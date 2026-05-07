"""
Adaptive training controller callback for Hugging Face Trainer.

At each evaluation step the callback emits one of three decisions:
  CONTINUE   — training is still improving; keep going.
  REDUCE_LR  — validation loss has plateaued; cut the learning rate and
                give the model a new patience window.
  STOP       — all four convergence conditions are simultaneously satisfied;
                training is terminated cleanly via TrainerControl.

Decision policy (per the specification):
  1. NEVER stop at the first plateau.
  2. Always try REDUCE_LR first (up to `max_lr_reductions` times).
  3. Only STOP when ALL of the following hold:
       a) Plateau persists across multiple patience windows.
       b) LR has already been reduced to its maximum allowed number of times.
       c) Gradient signal is weak or highly noisy.
       d) Marginal validation-loss improvement per training step is negligible.
"""

import logging
import math
from collections import deque
from typing import List, Optional, Tuple

from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(prev: Optional[float], new: float, alpha: float) -> float:
    """Exponential moving average. alpha=1 → no smoothing."""
    return new if prev is None else alpha * new + (1.0 - alpha) * prev


def _stats(values: List[float]):
    """Return (mean, std) for a non-empty list, else (None, None)."""
    if not values:
        return None, None
    n = len(values)
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / n
    return mean, math.sqrt(variance)


# ---------------------------------------------------------------------------
# Main callback
# ---------------------------------------------------------------------------

class TrainingControllerCallback(TrainerCallback):
    """
    Adaptive training controller that monitors validation loss, gradient
    statistics, and compute efficiency to decide whether to CONTINUE,
    REDUCE_LR, or STOP.

    Parameters
    ----------
    min_delta : float
        Minimum absolute decrease in (smoothed) validation loss to count
        as a meaningful improvement. Smaller improvements are ignored.
    patience : int
        Number of consecutive evaluation steps without meaningful improvement
        before an action is taken (REDUCE_LR or STOP evaluation).
    lr_reduction_factor : float
        Multiplicative factor applied to the learning rate on each REDUCE_LR
        action (e.g. 0.5 → halve the LR).
    max_lr_reductions : int
        Maximum number of REDUCE_LR actions allowed before the controller
        starts evaluating the STOP conditions.
    ema_alpha : float
        Smoothing coefficient for the exponential moving average of
        validation loss (range 0–1; lower = more smoothing).
    min_grad_norm : float
        Gradient norm below this value is treated as "weak signal".
    grad_snr_threshold : float
        If (mean_grad_norm / std_grad_norm) < this value the signal is
        considered "noisy / low SNR".
    grad_window : int
        Number of recent logged gradient norms to keep for statistics.
    min_marginal_improvement : float
        Minimum improvement in smoothed loss per training step to consider
        training budget-efficient.
    """

    CONTINUE  = "CONTINUE"
    REDUCE_LR = "REDUCE_LR"
    STOP      = "STOP"

    def __init__(
        self,
        min_delta: float = 1e-3,
        patience: int = 3,
        lr_reduction_factor: float = 0.5,
        max_lr_reductions: int = 3,
        ema_alpha: float = 0.1,
        min_grad_norm: float = 1e-2,
        grad_snr_threshold: float = 1.0,
        grad_window: int = 20,
        min_marginal_improvement: float = 1e-5,
    ):
        if not (0.0 < lr_reduction_factor < 1.0):
            raise ValueError("lr_reduction_factor must be in (0, 1).")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("ema_alpha must be in (0, 1].")

        self.min_delta = min_delta
        self.patience = patience
        self.lr_reduction_factor = lr_reduction_factor
        self.max_lr_reductions = max_lr_reductions
        self.ema_alpha = ema_alpha
        self.min_grad_norm = min_grad_norm
        self.grad_snr_threshold = grad_snr_threshold
        self.min_marginal_improvement = min_marginal_improvement

        # Running state (reset on train begin)
        self._reset_state()

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _reset_state(self):
        self._ema_loss: Optional[float] = None
        self._best_ema_loss: Optional[float] = None
        self._patience_counter: int = 0
        self._lr_reduction_count: int = 0
        self._eval_history: List[Tuple[int, float]] = []
        self._grad_norm_window: deque = deque(maxlen=20)
        self._last_known_lr: float = 0.0
        self.decision_log: List[dict] = []
        # Set by set_trainer(); None until then.
        self._trainer = None

    def set_trainer(self, trainer) -> None:
        """
        Call this after constructing the Trainer so the controller can access
        the optimizer directly.  The Trainer does not pass it through
        on_evaluate kwargs, so a back-reference is the cleanest solution.
        """
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        self._reset_state()
        logger.info("[TrainingController] Initialized and ready.")

    # ------------------------------------------------------------------
    # Gradient norm tracking (sourced from Trainer's periodic logs)
    # ------------------------------------------------------------------

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            try:
                self._grad_norm_window.append(float(grad_norm))
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Main decision point
    # ------------------------------------------------------------------

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            logger.debug("[TrainingController] No eval_loss in metrics; skipping.")
            return control

        step = state.global_step

        # Resolve optimizer and lr_scheduler from the stored trainer reference
        # (Trainer does not pass them through on_evaluate kwargs).
        optimizer    = getattr(self._trainer, "optimizer",    None) if self._trainer else None
        lr_scheduler = getattr(self._trainer, "lr_scheduler", None) if self._trainer else None

        # 1. Track current LR before any modification
        current_lr = self._read_lr(optimizer)
        self._last_known_lr = current_lr

        # 2. Smooth validation loss via EMA
        self._ema_loss = _ema(self._ema_loss, eval_loss, self.ema_alpha)
        smoothed = self._ema_loss
        self._eval_history.append((step, smoothed))

        # 3. Evaluate improvement against best observed smoothed loss
        if self._best_ema_loss is None:
            self._best_ema_loss = smoothed
            self._patience_counter = 0
            decision = self.CONTINUE
        else:
            improvement = self._best_ema_loss - smoothed
            if improvement > self.min_delta:
                # Meaningful improvement → reset patience, update best
                self._best_ema_loss = smoothed
                self._patience_counter = 0
                decision = self.CONTINUE
            else:
                # No meaningful improvement → increment patience
                self._patience_counter += 1
                decision = self._decide(optimizer, lr_scheduler, control)

        # 4. Log the decision
        self._record(step, eval_loss, smoothed, current_lr, decision)
        return control

    # ------------------------------------------------------------------
    # Decision logic
    # ------------------------------------------------------------------

    def _decide(
        self,
        optimizer,
        lr_scheduler,
        control: TrainerControl,
    ) -> str:
        # Patience window not yet exhausted → keep training
        if self._patience_counter < self.patience:
            return self.CONTINUE

        # Patience exhausted but LR reductions still available → REDUCE_LR
        if self._lr_reduction_count < self.max_lr_reductions:
            new_lr = self._apply_lr_reduction(optimizer, lr_scheduler)
            self._lr_reduction_count += 1
            self._patience_counter = 0   # fresh window after reduction
            logger.warning(
                f"[TrainingController] REDUCE_LR #{self._lr_reduction_count}"
                f"/{self.max_lr_reductions}: LR → {new_lr:.3e}"
            )
            return self.REDUCE_LR

        # All LR reductions exhausted → check the four STOP conditions
        cond_plateau   = self._patience_counter >= self.patience            # (a)
        cond_lr_low    = self._lr_reduction_count >= self.max_lr_reductions # (b)
        cond_grad_weak = self._grad_signal_weak()                           # (c)
        cond_marginal  = self._marginal_improvement_negligible()            # (d)

        if cond_plateau and cond_lr_low and cond_grad_weak and cond_marginal:
            control.should_training_stop = True
            logger.warning(
                "[TrainingController] STOP: all convergence conditions met. "
                f"patience_counter={self._patience_counter}, "
                f"lr_reductions={self._lr_reduction_count}, "
                f"grad_weak={cond_grad_weak}, "
                f"marginal_negligible={cond_marginal}."
            )
            return self.STOP
        else:
            # Not all four conditions met → CONTINUE but warn
            logger.warning(
                "[TrainingController] CONTINUE (post max-LR-reductions). "
                f"Conditions — plateau={cond_plateau}, lr_low={cond_lr_low}, "
                f"grad_weak={cond_grad_weak}, marginal_neg={cond_marginal}. "
                "Waiting until all four are satisfied to STOP."
            )
            return self.CONTINUE

    # ------------------------------------------------------------------
    # Condition evaluators
    # ------------------------------------------------------------------

    def _grad_signal_weak(self) -> bool:
        """
        Returns True if the gradient norm is very small OR the signal-to-noise
        ratio (mean / std) of recent gradient norms is below the threshold.
        Requires at least 5 data points; returns False otherwise (cautious).
        """
        history = list(self._grad_norm_window)
        if len(history) < 5:
            # Not enough gradient data — assume healthy to avoid false stops
            logger.debug(
                "[TrainingController] Insufficient grad norm history "
                f"({len(history)} samples); treating signal as healthy."
            )
            return False

        mean, std = _stats(history)

        if mean < self.min_grad_norm:
            logger.debug(f"[TrainingController] Grad norm mean {mean:.4e} < threshold {self.min_grad_norm:.4e}.")
            return True

        snr = mean / (std + 1e-9)
        if snr < self.grad_snr_threshold:
            logger.debug(
                f"[TrainingController] Grad SNR {snr:.3f} < threshold {self.grad_snr_threshold:.3f} "
                f"(mean={mean:.4e}, std={std:.4e})."
            )
            return True

        return False

    def _marginal_improvement_negligible(self) -> bool:
        """
        Returns True if the average improvement in smoothed loss per training
        step, computed over the second half of evaluation history, is below
        `min_marginal_improvement`. Requires at least 4 eval points.
        """
        if len(self._eval_history) < 4:
            return False

        half = len(self._eval_history) // 2
        recent = self._eval_history[-half:]

        if len(recent) < 2:
            return False

        first_step, first_loss = recent[0]
        last_step,  last_loss  = recent[-1]
        step_span = last_step - first_step

        if step_span <= 0:
            return False

        # Positive → improving; negative → diverging
        marginal = (first_loss - last_loss) / step_span
        is_negligible = marginal < self.min_marginal_improvement
        logger.debug(
            f"[TrainingController] Marginal improvement/step={marginal:.2e}, "
            f"threshold={self.min_marginal_improvement:.2e}, "
            f"negligible={is_negligible}."
        )
        return is_negligible

    # ------------------------------------------------------------------
    # Learning rate helpers
    # ------------------------------------------------------------------

    def _read_lr(self, optimizer) -> float:
        if optimizer is not None and hasattr(optimizer, "param_groups") and optimizer.param_groups:
            return float(optimizer.param_groups[0]["lr"])
        return self._last_known_lr

    def _apply_lr_reduction(self, optimizer, lr_scheduler) -> float:
        """
        Reduce the LR by `lr_reduction_factor` in the optimizer, and scale
        the scheduler's base LRs by the same factor so the cosine (or other)
        schedule continues from the new baseline rather than fighting the
        manual reduction.
        """
        if optimizer is None:
            logger.warning(
                "[TrainingController] Optimizer not available in callback kwargs; "
                "LR reduction could not be applied."
            )
            return self._last_known_lr

        for pg in optimizer.param_groups:
            pg["lr"] *= self.lr_reduction_factor

        # Scale the scheduler's base_lrs so the schedule's future values
        # are computed relative to the reduced LR.
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs = [
                b * self.lr_reduction_factor for b in lr_scheduler.base_lrs
            ]

        new_lr = float(optimizer.param_groups[0]["lr"])
        self._last_known_lr = new_lr
        return new_lr

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _record(
        self,
        step: int,
        raw_loss: float,
        ema_loss: float,
        lr: float,
        decision: str,
    ):
        entry = {
            "step":         step,
            "raw_loss":     round(raw_loss,  6),
            "ema_loss":     round(ema_loss,  6),
            "lr":           lr,
            "patience":     self._patience_counter,
            "lr_reductions": self._lr_reduction_count,
            "decision":     decision,
        }
        self.decision_log.append(entry)

        level = logging.WARNING if decision != self.CONTINUE else logging.INFO
        logger.log(
            level,
            "[TrainingController] step=%-8d  raw_loss=%.6f  ema_loss=%.6f  "
            "lr=%.3e  patience=%d/%d  lr_reductions=%d/%d  → %s",
            step,
            raw_loss, ema_loss, lr,
            self._patience_counter, self.patience,
            self._lr_reduction_count, self.max_lr_reductions,
            decision,
        )

    # ------------------------------------------------------------------
    # Public utility
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of all decisions made."""
        if not self.decision_log:
            return "No evaluation steps recorded."
        lines = ["TrainingController decision log:"]
        for e in self.decision_log:
            lines.append(
                f"  step={e['step']:>8}  raw={e['raw_loss']:.6f}  "
                f"ema={e['ema_loss']:.6f}  lr={e['lr']:.3e}  "
                f"patience={e['patience']}  reductions={e['lr_reductions']}  "
                f"→ {e['decision']}"
            )
        return "\n".join(lines)

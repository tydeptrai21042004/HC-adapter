# models/tuning_modules/__init__.py  (excerpt)

from .prompter import PadPrompter
from .conv_adapter import ConvAdapter, LinearAdapter
from .program_module import ProgramModule

def set_tuning_config(tuning_method, args):
    """
    Build a small config dict for the chosen tuning method.
    Backward-compatible with legacy names and adds:
      - 'hcc'        → Hartley–Cosine Adapter
      - 'residual'   → Residual Adapters (parallel/series)

    Supported (legacy & aliases):
      conv_adapt | conv_adapt_norm | conv_adapt_bias | conv | conv_adapter
      prompt
      full | linear | norm | repnet | repnet_bias | bias
      hcc | hcc_adapter
      residual | residual_adapter | residual_adapters | ra
    """
    # ---- normalize a few common aliases (keeps old strings working) ----------
    alias = {
        "conv": "conv_adapt",
        "conv-adapter": "conv_adapt",
        "conv_adapter": "conv_adapt",

        "hcc_adapter": "hcc",

        "residual_adapter": "residual",
        "residual_adapters": "residual",
        "ra": "residual",
    }
    tm = alias.get(tuning_method, tuning_method)

    # ---- Conv-Adapter family (unchanged behavior) ----------------------------
    if tm in ("conv_adapt", "conv_adapt_norm", "conv_adapt_bias"):
        return {
            "method": tm,  # keep exact string for downstream builder
            "kernel_size": getattr(args, "kernel_size", 3),
            "adapt_size": getattr(args, "adapt_size", 8),
            "adapt_scale": getattr(args, "adapt_scale", 1.0),
        }

    # ---- Prompt (unchanged) --------------------------------------------------
    if tm == "prompt":
        return {
            "method": tm,
            "prompt_size": getattr(args, "prompt_size", 10),
        }

    # ---- Simple switches (unchanged) -----------------------------------------
    if tm in ("full", "linear", "norm", "repnet", "repnet_bias", "bias"):
        return {"method": tm}

    # ---- NEW: Hartley–Cosine Adapter (HCC) -----------------------------------
    if tm == "hcc":
        return {
            "method": "hcc",
            # matches your HCCAdapter.__init__ signature
            "M":              getattr(args, "hcc_M", 1),
            "h":              getattr(args, "hcc_h", 1),
            "axis":           getattr(args, "hcc_axis", "hw"),
            "per_channel":    getattr(args, "hcc_per_channel", True),
            "tie_sym":        getattr(args, "hcc_tie_sym", True),
            "use_pw":         getattr(args, "hcc_use_pw", True),
            "pw_ratio":       getattr(args, "hcc_pw_ratio", 8),
            "use_bn":         getattr(args, "hcc_use_bn", True),
            "residual_scale": getattr(args, "hcc_residual_scale", 1.0),
            "gate_init":      getattr(args, "hcc_gate_init", 0.1),
            "padding_mode":   getattr(args, "hcc_padding", "reflect"),
        }

    # ---- NEW: Residual Adapters (parallel/series) ----------------------------
    if tm == "residual":
        return {
            "method": "residual",
            "mode":        getattr(args, "ra_mode", "parallel"),      # "parallel" | "series"
            "reduction":   getattr(args, "ra_reduction", 16),         # bottleneck ratio
            "norm":        getattr(args, "ra_norm", "bn"),            # "bn" | "ln" | "none"
            "act":         getattr(args, "ra_act", "relu"),           # "relu" | "gelu" | "silu" | "none"
            "gate_init":   getattr(args, "ra_gate_init", 0.0),
            "stages":      getattr(args, "ra_stages", "1,2,3,4"),     # e.g., "2,3,4"
        }

    # ---- otherwise ------------------------------------------------------------
    raise NotImplementedError(f"Unknown tuning_method: {tuning_method}")

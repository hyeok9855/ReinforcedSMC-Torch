import argparse

from energies import BaseEnergy

from .base import BaseModule
from .ddsmlp_modules import DDSMLPModule
from .mlp_modules import MLPModule
from .pismlp_modules import PISMLPModule


def get_module(args: argparse.Namespace, energy: BaseEnergy) -> BaseModule:
    common_kwargs = {
        "conditional_flow_model": args.conditional_flow_model,
        "lp": args.lp,
        "lp_scaling_per_dimension": args.lp_scaling_per_dimension,
        "clipping": args.clipping,
        "out_clip": args.out_clip,
        "lgv_clip": args.lgv_clip,
        "learn_pb": args.learn_pb,
        "pb_scale_range": args.pb_scale_range,
        "learn_variance": args.learn_variance,
        "log_var_range": args.log_var_range,
        "use_checkpoint": args.use_checkpoint,
    }
    if "mlp" in args.module:
        mlp_kwargs = {
            "ndim": energy.ndim,
            "harmonics_dim": args.hidden_dim,
            "t_emb_dim": args.hidden_dim,
            "s_emb_dim": args.hidden_dim,
            "hidden_dim": args.hidden_dim,
            "joint_layers": args.joint_layers,
            "zero_init": args.zero_init,
            "share_embeddings": args.share_embeddings,
            "flow_harmonics_dim": args.flow_hidden_dim,
            "flow_t_emb_dim": args.flow_hidden_dim,
            "flow_s_emb_dim": args.flow_hidden_dim,
            "flow_hidden_dim": args.flow_hidden_dim,
            "flow_layers": args.flow_layers,
            "lgv_layers": args.lgv_layers,
        }

        module_cls = {
            "mlp": MLPModule,
            "pismlp": PISMLPModule,
            "ddsmlp": DDSMLPModule,
        }[args.module]
        return module_cls(**common_kwargs, **mlp_kwargs)

    else:
        raise ValueError(f"Module {args.module} not found")

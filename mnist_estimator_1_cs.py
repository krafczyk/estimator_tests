"""
Script to train MNIST model.
"""


import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import os


#sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.environ['HOME'])


from modelzoo.common.tf.estimator.cs_estimator import CerebrasEstimator
from cerebras.tf.cs_model_to_estimator import KerasModelToCerebrasEstimator
from modelzoo.common.tf.estimator.run_config import CSRunConfig
from modelzoo.common.tf.estimator.utils import (
    cs_disable_summaries,
    cs_enable_summaries,
)
from modelzoo.common.tf.run_utils import (
    check_env,
    get_csrunconfig_dict,
    is_cs,
    save_params,
    update_params_from_args,
)
from modelzoo.fc_mnist.tf.utils import (
    get_custom_stack_params,
    get_params,
)


from mnist_estimator_1_lib import test_input_fn, train_input_fn, model_fn, build_model_fn
import data


_curdir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_YAML_PATH = os.path.join(_curdir, "params_1.yaml")


def create_arg_parser(default_model_dir):
    """
    Create parser for command line args.

    :param str default_model_dir: default value for the model_dir
    :returns: ArgumentParser
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        default=DEFAULT_YAML_PATH,
        help="Path to .yaml file with model parameters",
    )
    parser.add_argument(
        "-o",
        "--model_dir",
        default=default_model_dir,
        help="Model directory where checkpoints will be written. "
        + "If directory exists, weights are loaded from the checkpoint file.",
    )
    parser.add_argument(
        "--cs_ip",
        default=None,
        help="IP address of the Cerebras System, defaults to None. Ignored on GPU.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=(
            "Number of steps to run mode train."
            + " Runs repeatedly for the specified number."
        ),
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode train or for defining training"
            + " configuration for train_and_eval. Runs incrementally till"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help=(
            "Number of total steps to run mode eval, eval_all or for defining"
            + " eval configuration for train_and_eval. Runs once for"
            + " the specified number."
        ),
    )
    parser.add_argument(
        "-m",
        "--mode",
        required=True,
        choices=["train", "eval", "eval_all", "train_and_eval"],
        help=(
            "Can train, eval, eval_all, or train_and_eval."
            + "  Train and eval will compile and train if on the Cerebras System,"
            + "  and just run locally (CPU/GPU) if not on the Cerebras System."
            + "  train_and_eval will run locally."
            + "  Eval_all will run eval locally for all available checkpoints."
        ),
    )
    parser.add_argument(
        "--multireplica",
        action="store_true",
        help="run multiple copies of the model data-parallel"
        + " on the wafer at the same time.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Compile model up to kernel matching.",
    )
    parser.add_argument(
        "--compile_only",
        action="store_true",
        help="Compile model completely, generating compiled executables.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force model to run on a specific device (e.g., --device /gpu:0)",
    )

    return parser


def validate_params(params):
    # check validate_only/compile_only
    runconfig_params = params["runconfig"]
    assert not (
        runconfig_params["validate_only"] and runconfig_params["compile_only"]
    ), "Please only use one of validate_only and compile_only."

    # ensure runconfig is compatible with the Cerebras System
    if (
        is_cs(runconfig_params)
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        assert (
            runconfig_params["mode"] == "train"
        ), "For FC_MNIST model, only training is supported on the Cerebras System."
    else:
        assert not runconfig_params[
            "multireplica"
        ], "Multi-replica training is only possible on the Cerebras System."


def run(
    args, params, model_fn, test_input_fn=None, train_input_fn=None):
    """
    Set up estimator and run based on mode

    :params dict params: dict to handle all parameters
    :params tf.estimator.EstimatorSpec model_fn: Model function to run with
    :params callable input_fn: Input function to use
    """
    # update and validate runtime params
    runconfig_params = params["runconfig"]
    update_params_from_args(args, runconfig_params)
    validate_params(params)
    # save params for reproducibility
    save_params(params, model_dir=runconfig_params["model_dir"])

    # get runtime configurations
    use_cs = is_cs(runconfig_params)
    csrunconfig_dict = get_csrunconfig_dict(runconfig_params)
    stack_params = get_custom_stack_params(params)

    if (
        use_cs
        or runconfig_params["validate_only"]
        or runconfig_params["compile_only"]
    ):
        try:
            from cerebras.pb.stack.autogen_pb2 import AP_ENABLED
            from cerebras.pb.stack.full_pb2 import FullConfig
            stack_params["config"] = FullConfig()
            stack_params['config'].matching.autogen_policy = AP_ENABLED
        except ImportError:
            pass

    # prep cs1 run environment, run config and estimator
    check_env(runconfig_params)
    est_config = CSRunConfig(
        cs_ip=runconfig_params["cs_ip"],
        stack_params=stack_params,
        **csrunconfig_dict,
    )

    est = CerebrasEstimator(
        model_fn=model_fn,
        model_dir=runconfig_params["model_dir"],
        config=est_config,
        params=params,
    )

    #est = KerasModelToCerebrasEstimator(
    #    model_fn=build_model_fn(),
    #    model_dir=runconfig_params["model_dir"],
    #    config=est_config,
    #    params=params,
    #)

    # execute based on mode
    if runconfig_params["validate_only"] or runconfig_params["compile_only"]:
        if runconfig_params["mode"] == "train":
            mode=tf.estimator.ModeKeys.TRAIN
            est.compile(
                train_input_fn, validate_only=runconfig_params["validate_only"], mode=mode
            )
        else:
            mode = tf.estimator.ModeKeys.EVAL
            est.compile(
                test_input_fn, validate_only=runconfig_params["validate_only"], mode=mode
            )
    elif runconfig_params["mode"] == "train":
        est.train(
            input_fn=train_input_fn,
            steps=runconfig_params["steps"],
            max_steps=runconfig_params["max_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval":
        est.evaluate(
            input_fn=test_input_fn,
            steps=runconfig_params["eval_steps"],
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "predict":
        est.predict(
            input_fn=test_input_fn,
            use_cs=use_cs,
        )
    elif runconfig_params["mode"] == "eval_all":
        ckpt_list = tf.train.get_checkpoint_state(
            runconfig_params["model_dir"]
        ).all_model_checkpoint_paths
        for ckpt in ckpt_list:
            est.evaluate(
                input_fn=test_input_fn,
                checkpoint_path=ckpt,
                steps=runconfig_params["eval_steps"],
                use_cs=use_cs,
            )
    elif runconfig_params["mode"] == "train_and_eval":
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=runconfig_params["max_steps"],
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=test_input_fn,
            steps=runconfig_params["eval_steps"],
            throttle_secs=runconfig_params["throttle_secs"],
        )
        tf.estimator.train_and_evaluate(est, train_spec, eval_spec)


def main():
    """
    Main function
    """
    default_model_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model_dir"
    )
    parser = create_arg_parser(default_model_dir)
    args = parser.parse_args(sys.argv[1:])
    params = get_params(args.params)
    summary_context = (
        cs_disable_summaries if args.multireplica else cs_enable_summaries
    )
    with summary_context():
        run(
            args=args,
            params=params,
            model_fn=model_fn,
            #test_input_fn=test_input_fn,
            test_input_fn=data.eval_input_fn,
            train_input_fn=data.train_input_fn,
        )


if __name__ == "__main__":
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    main()

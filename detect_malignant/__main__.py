import abc
import json
import ssl

from detect_malignant.src.configs.config import ExperimentationConfig
# from detect_malignant.finetune import finetune
from detect_malignant.test.test import test
from detect_malignant.train import train


ssl._create_default_https_context = ssl._create_unverified_context

class Executor:
    def __init__(self, args):
        print(args.config)
        if args.config is not None:
            with open(args.config) as json_file:
                self.config = json.load(json_file)

            if "mode" in args and ("mode" not in self.config or self.config["mode"] != args.mode):
                self.config["mode"] = args.mode

            if args.batch_size:
                self.config["batch_size"] = args.batch_size
        else:
            self.config = None
        self.args = args

    @abc.abstractmethod
    def execute(self):
        return NotImplemented


class TrainingExecutor(Executor):
    def __init__(self, args):
        super(TrainingExecutor, self).__init__(args)
        assert self.config is not None, "need configuration file for training provided"
        self.config["mode"] = args.mode  # update the mode value from config

    def execute(self):
        train_settings = ExperimentationConfig.parse_obj(self.config)
        train(train_settings)


# class FinetuneExecutor(Executor):
#     def __init__(self, args):
#         super(FinetuneExecutor, self).__init__(args)
#         assert self.config is not None, "need configuration file for training provided"
#         self.config["mode"] = args.mode  # update the mode value from config

#     def execute(self):
#         finetune_settings = ExperimentationConfig.parse_obj(self.config)
#         finetune(finetune_settings)


class TestingExecutor(Executor):
    def __init__(self, args):
        super(TestingExecutor, self).__init__(args)
        assert self.config is None, "No configuration needed for testing"

    def execute(self):  
        test(self.args.exp_dir)




EXECUTORS = {
    "train": TrainingExecutor,
    "test": TestingExecutor,
    # "finetune": FinetuneExecutor,
}


def get_executor(mode: str) -> Executor:
    return EXECUTORS[mode]


def run(args=None):
    from argparse import ArgumentParser

    parser = ArgumentParser("detect_malignant")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=sorted(EXECUTORS.keys()),
        help="Overwrite mode from the configuration file",
    )

    parser.add_argument("--config", type=str, help="The configuration file for training")
    parser.add_argument("--exp-dir", type=str, help="The experiment directory for tests")
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Overwrite the batch size from configuration",
        default=None,
              )


    args = parser.parse_args(args)
    ExecutorClass = get_executor(args.mode)
    executor = ExecutorClass(args)
    print("executor object:", executor)  # Add this line to print the executor object
    executor.execute()


if __name__ == "__main__":
    run()
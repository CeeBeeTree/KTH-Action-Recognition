from KTH_DataModule import KTH_DataModule
from KTH_VideoBlockClassifier import KTH_VideoBlockClassifier
import pytorch_lightning as pl
from argparse import ArgumentParser

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data", dest="data_path")
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data_module = KTH_DataModule(args.data_path)

    # train
    model = KTH_VideoBlockClassifier()
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, data_module)

    # evaluate
    trainer.test(model,data_module)
from KTH_DataModule import KTH_DataModule
from KTH_VideoBlockClassifier import KTH_VideoBlockClassifier
import pytorch_lightning as pl
from argparse import ArgumentParser

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/data", dest="data_path")
    parser.add_argument("--use_cached_data", type=bool, default=FALSE, dest="use_preloaded")
     
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    data_module = KTH_DataModule(directory=args.data_path, 
                                 use_preloaded=args.use_preloaded)

    # train
    model = KTH_VideoBlockClassifier()
    trainer = pl.Trainer.from_argparse_args(args)

    trainer.fit(model, data_module)

    # evaluate
    trainer.test(model,data_module)
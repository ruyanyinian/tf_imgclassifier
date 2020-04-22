from core.tfrecord_generation import TfGen
from core.config import cfg

if __name__ == '__main__':
    tfgen = TfGen("train",
                  img_dir=cfg.DATA.TEST_DIR,
                  output_dir=cfg.DATA.TEST_OUTPUT_DIR,
                  label_txt_path=cfg.DATA.LABEL_FILEPATH,
                  shard_nums=cfg.DATA.TEST_SHARDS,
                  thread_nums=cfg.DATA.THREADS)

    tfgen.img_info(cfg.LOGS.INFO_TEST)

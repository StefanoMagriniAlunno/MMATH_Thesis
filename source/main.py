from packages import common, synthesis

logger = common.main("logs/dev.log")
synthesis.synth(
    logger,
    "data/db/images_pgm",
    "data/out",
    "temp/list_file.txt",
    "logs/synthesis.log",
    8,
)

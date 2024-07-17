from packages import common, synthesis

logger = common.main("logs/dev.log")
synthesis.synth(
    logger, "data/db/images", "data/out", "temp/list_file.txt", "logs/synthesis.log"
)

# -*- coding: utf-8 -*-

import logging
import sys

logger = logging.getLogger("Summarization logger")

formatter = logging.Formatter('%(asctime)s %(levelname)-8s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.formatter = formatter
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

logger.setLevel(logging.DEBUG)


def log_score(scores_all):
    # res = "Rouge1:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    #     scores_all['rouge-1']['p'], scores_all['rouge-1']['r'], scores_all['rouge-1']['f']) \
    #       + "Rouge2:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    #           scores_all['rouge-2']['p'], scores_all['rouge-2']['r'], scores_all['rouge-2']['f']) \
    #       + "Rougel:\n\tp:%.6f, r:%.6f, f:%.6f\n" % (
    #           scores_all['rouge-l']['p'], scores_all['rouge-l']['r'], scores_all['rouge-l']['f'])
    #
    res = f"""
        ROUGE_1:
            p={scores_all['rouge-1']['p']}
            r={scores_all['rouge-1']['r']}
            f={scores_all['rouge-1']['f']}
        ROUGE_2:
            p={scores_all['rouge-2']['p']}
            r={scores_all['rouge-2']['r']}
            f={scores_all['rouge-2']['f']}
        ROUGE_L:
            p={scores_all['rouge-l']['p']}
            r={scores_all['rouge-l']['r']}
            f={scores_all['rouge-l']['f']}
    """
    logger.info(res)

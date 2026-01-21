import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from multiprocessing import current_process


def set_logger(log_dir: str = './logs'):	
    # 현지 시각 기준으로 타임스탬프 설정
    kst = timezone(timedelta(hours=9))
    now = datetime.now(kst)
    timestamp = now.strftime('%Y%m%d_%H%M%S')

    logger = logging.getLogger('svdfm_test')
    logger.setLevel(logging.INFO)

    # 중복 방지
    if logger.handlers:
        return logger
    logger.propagate = False

    rank_env = os.environ.get('LOCAL_RANK') or os.environ.get('RANK') or os.environ.get('GLOBAL_RANK')
    rank = int(rank_env) if rank_env is not None else 0

    # 포매팅
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y%m%d %H:%M:%S')

    if current_process().name == 'MainProcess' and rank==0:
        # 로그 저장 폴더
        os.makedirs(log_dir, exist_ok=True)
        log_filepath = os.path.join(log_dir, f'{timestamp}.log')
        # 파일 핸들러
        fh = logging.FileHandler(log_filepath)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        # 콘솔 핸들러 (터미널 출력)
        ch = logging.StreamHandler(sys.stdout)
        logger.addHandler(ch)

    return logger

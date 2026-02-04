import logging
import os
from datetime import datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def setup_logging(level=logging.INFO) -> None:
    logs_dir = os.path.join(ROOT, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    logging.getLogger().handlers.clear()
    
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(
        filename=os.path.join(ROOT, "logs", f"log_{time}.log"), 
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)
from multiprocessing import Pool, Queue
import time
import os
import sounddevice as sd

# Worker function
def square(x):
    print(f"[Worker {os.getpid()}] Computing square({x})")
    time.sleep(2)  # simulate heavy work
    return x * x

# Callback to handle result

    
def a(pool, x):
    def handle_result(result):
        out.put(result)
        print(f"[Main] Got result: {result}")
    def audio_callback(indata, frames, time, status):
        pool.apply_async(square, args=(x,), callback=handle_result)
    with sd.InputStream(callback=audio_callback):
        print("[Main] Starting async computation...")
        time.sleep(5)

if __name__ == "__main__":
    # Create a pool with 2 workers
    x = 2
    out = Queue()
    with Pool(processes=10) as pool:
        try:
            for i in range(4):
                a(pool, x)
        except:
            pass
        # Close and wait for completion
        pool.close()
        pool.join()
    while not out.empty():
        print(out.get())

    print("[Main] Done!")



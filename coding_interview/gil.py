import multiprocessing
import asyncio
import aiohttp
from pathlib import Path


def compute_square(numbers: list[int], result: list[int], index: int):
    result[index] = sum([n**2 for n in numbers])

def multi_process(n_processes: int = 4, upper_bound: int = 100000):
    """This example uses multiprocessing to calculate the sum of squares of numbers in parallel.
      Each process works independently with shared memory."""
    numbers = list(range(upper_bound))
    result = multiprocessing.Array('i', n_processes)
    chunk_size = upper_bound // n_processes

    processes = []
    for i in range(n_processes):
        start_index = i * chunk_size
        end_index = (i+1) * chunk_size 
        p = multiprocessing.Process(target=compute_square, args=(numbers[start_index:end_index], result, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    print(result[:])


async def download_image(session, url: str, save_path: str | Path):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.read()
                with open(save_path, 'wb') as f:
                    f.write(content)
                    print(f"Download {url}")
            else:
                print(f'Download failed with status {response.status}') 
    except Exception as e:
        print(f"Print {url}: {e}")

async def download_all_images(url_list: list[str], save_path: str | Path, max_threads: int = 32):
    semaphore = asyncio.Semaphore(max_threads)
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, url in enumerate(url_list):
            task = download_image_with_semaphore(semaphore, session, url, save_path / f'image_{i}.jpg')
            tasks.append(task)
        await asyncio.gather(*tasks)
    print("Finished download_all_images")

async def download_image_with_semaphore(semaphore, session, url: str, save_path: str | Path):
    async with semaphore:
        await download_image(session, url, save_path)

def multi_thread():
    url_list = ["https://images.unsplash.com/photo-1497206365907-f5e630693df0?q=80&w=2080&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1486365227551-f3f90034a57c?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://unsplash.com/photos/brown-bear-selective-focal-photo-during-daytime-aRXPJnXQ9lU"]
    print("Calling asyncio.run")
    asyncio.run(download_all_images(url_list, Path('.'), 5))
    print("Finished asyncio.run")



if __name__=='__main__':
    #multi_process(5, 1000000)
    multi_thread()
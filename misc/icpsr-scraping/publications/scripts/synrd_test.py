import synrd_functions as synrd


if __name__ == "__main__":
    synrd.setup()
    print("Testing single...")
    print(synrd.test_single_icpsr(rows=5))
    print("")
    print("Testing threaded...")
    synrd.test_threaded_icpsr(num_rows=5, num_pages=5)

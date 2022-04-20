import synrd_functions as synrd


if __name__ == "__main__":
    synrd.setup()
    print("Actual threaded processing...")
    print(synrd.run_icpsr(start=90000))

import glob


def main():
    paths = glob.glob("data/NSDUH_Versions/*/*/*.tsv") + glob.glob(
        "data/NSDUH_Versions/*/*/*/*.tsv"
    )

    delim = "\t"
    for path in paths:
        if "Tab" in path:
            with open(path, "r") as f:
                n_head = next(f).count(delim)
                n_cont = next(f).count(delim)
                if n_head != n_cont:
                    print(
                        f"{path.split('/')[-1]}, number of columns in header: {n_head}, number of columns in the 1st row: {n_cont}"
                    )


if __name__ == "__main__":
    main()

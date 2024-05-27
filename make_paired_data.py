import os

def main():
    all_cloth = os.listdir("./datasets/train/cloth")
    with open ("./datasets/train_pairs.txt", "w") as f:
        for i in range(len(all_cloth)-1):
            f.write(f"{all_cloth[i]} {all_cloth[i+1]}\n")

if __name__ == "__main__":
    main()
import argparse
from src.haar import HaarCascadeClassifier
from src.optimizer import HaarOptimizer


def run_task(args):
    if args.task == "detect":
        clf = HaarCascadeClassifier()
        clf.classify(args.image)
        return None

    if args.task == "optimize":
        optimizer = HaarOptimizer(args.dataset)
        optimizer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("task", nargs='?', type=str, default="detect")
    parser.add_argument("--image", '-i', help="Path to an input image")
    parser.add_argument("--dataset", '-d', help="Path to a dataset directory")

    args = parser.parse_args()

    if not args.image and args.task == "detect":
        print("ERROR: Please provide an input image with the -i option")
        raise SystemExit(1)

    if not args.dataset and args.task == "optimize":
        print("ERROR: Please provide a path to a dataset directory")
        raise SystemExit(1)

    run_task(args)

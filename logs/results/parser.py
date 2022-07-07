import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename',
                        dest = "filename",
                        help = "filename",
                        default = "metrics_10_0.05_100_dc.log",
                        required = True)
    return parser.parse_args()


def get_average(filename):
    tot_average_train_loss = 0.0
    tot_test_loss = 0.0
    tot_test_accuracy = 0.0
    tot_asr = 0.0

    peers = 0
    with open(filename, "r") as f:
        content = f.readlines()
        for line in content:
            line = line.strip()
            line = line.split(",")
            average_train_loss = float(line[1])
            test_loss = float(line[2])
            test_accuracy = float(line[3])
            asr = float(line[4])
            peers += 1
            tot_average_train_loss += average_train_loss
            tot_test_loss += test_loss
            tot_test_accuracy += test_accuracy
            tot_asr += asr
    
    print("Train Loss = {}, Test Loss = {}, Test Accuracy = {}, ASR = {}".format(str(tot_average_train_loss/peers), str(tot_test_loss/peers), str(tot_test_accuracy/peers), str(tot_asr/peers)))


def main():
    print('parse results')

    filename = parse_args().filename
    get_average(filename)


if __name__ == '__main__':
    main()
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--filename',
                        dest = "filename",
                        help = "filename",
                        default = "metrics_10_0.05_100_dc.log",
                        required = True)
    return parser.parse_args()

dishonest_peers = ['73', '26', '90', '82', '21', '38', '75', '29', '34', '41', '20', '45', '22', '1', '95', '55', '88', '85', '83', '36', '71', '43', '15', '84', '33', '3', '32', '52', '77', '49', '8', '89', '98', '62', '59', '7', '4', '86', '46', '99', '81', '42', '27', '80', '19', '87', '74', '17', '9', '58']


def get_peer_id(peer):
    pid = peer.split(".")
    pid = pid[0].split("r")
    return pid[-1]


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
            peer = line[0]
            if peer not in dishonest_peers:
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
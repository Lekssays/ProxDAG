import model
import utils


def main():
    print("Models Updates Verifier")

    net = model.Net()
    
    sample = net.state_dict()
    content_bytes = utils.to_bytes(sample)

    modelID = "some id"
    parents = ["parent1", "parent2"]
    endpoint = "peer1.org1.example.com"
    model_update_pb = utils.to_protobuf(
        modelID=modelID,
        parents=parents,
        content=content_bytes,
        endpoint=endpoint
    )

    content = utils.from_bytes(model_update_pb.content)
    print(content)

    messageID = utils.send_model_update(model_update_pb)
    print(messageID)

if __name__ == '__main__':
    main()
import torch

def load_data(file_name):
    with open(file_name, 'r') as file:
        data = file.read().splitlines()

    loaded_data = []
    max_frame_id = 0

    for item in data:
        frame_id, track_id, x, y, w, h, c, _, _ = list(map(float, item.split(',')))
        frame_id = int(frame_id)
        track_id = int(track_id)
        c = int(c)

        if c != 1:
            continue

        loaded_data.append([frame_id, track_id, x, y, w, h])

        if frame_id > max_frame_id:
            max_frame_id = frame_id

    data_grouped_by_track = dict()
    for item in loaded_data:
        frame_id, track_id, x, y, w, h = item

        if track_id not in data_grouped_by_track:
            data_grouped_by_track[track_id] = []

        data_grouped_by_track[track_id].append([frame_id, x, y, w, h])

    inputs = [[] for _ in range(max_frame_id)]
    outputs = [[] for _ in range(max_frame_id)]

    for track_id, data in data_grouped_by_track.items():
        for i in range(len(data) - 2):
            _, x1, y1, w1, h1 = data[i]
            frame_id, x2, y2, w2, h2 = data[i + 1]
            _, x3, y3, w3, h3 = data[i + 2]

            inputs[frame_id - 1].append([x2, y2, w2, h2, x2 - x1, y2 - y1, w2 - w1, h2 - h1])
            outputs[frame_id - 1].append([x3 - x2, y3 - y2, w3 - w2, h3 - h2])

    inputs = list(map(lambda x: torch.tensor(x), inputs[2:]))
    outputs = list(map(lambda x: torch.tensor(x), outputs[2:]))

    return inputs, outputs

if __name__ == "__main__":
    load_data("gt.txt")

from torch.utils.data import Dataset, DataLoader
import torch
import AudioData


class dataset(Dataset):
    def __init__(self, mix_reader, target_readers):
        super(dataset).__init__()
        self.mix_reader = mix_reader
        self.target_readers = target_readers
        self.keys = mix_reader.wave_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        if key not in self.keys:
            raise ValueError
        return (self.mix_reader[key], [target[key] for target in self.target_readers])


class dataloader(object):
    def __init__(self, dataset, batch_size=2, shuffle=True, num_workers=1):
        super(dataloader).__init__()
        self.dataload = DataLoader(
            dataset, batch_size=batch_size, num_workers=1, shuffle=shuffle, collate_fn=self.collate)

    def transform(self, mix_wave, target_waves):
        times = mix_wave[0]

    def collate(self, batchs):
        print(1)
        mix_wave = batchs[0]
        target_waves = batchs[1]

    def __iter__(self):
        for b in self.dataload:
            yield b


if __name__ == "__main__":
    mix_reader = AudioData.AudioData(
        "/home/likai/data1/create_scp/cv_mix.scp", is_mag=True)
    target_readers = [AudioData.AudioData("/home/likai/data1/create_scp/cv_s1.scp", is_mag=True),
                      AudioData.AudioData("/home/likai/data1/create_scp/cv_s2.scp", is_mag=True), ]
    dataset = dataset(mix_reader, target_readers)
    dataloader = dataloader(dataset)
    for i in dataloader:
        print(i)

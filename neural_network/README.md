## Classes

```mermaid
graph TD
    Fig0[EqualGridFigure]
    Anime0>anime.py]
    Base0((BaseData))
    Raw0[RawG1000Data]
    Raw1[RawPhoneData]
    Raw2[RawStratuxData]
    Flight0[FlightExtensionData]
    Flight1[FlightPruneData]
    BasePair0((BasePairData))
    SeqPair0[FlightSequencePairData]
    Norm0[NormalizationData]
    FramePair0[FlightFramePairData]
    Batch0[FlightPairDataLoader]
    Iter0[FlightPairDataLoaderIterator]
    Batch1[TimestepPairDataLoader]
    Iter1[TimestepPairDataLoaderIterator]
    Manager0[NeuralNetworkManager]
    style SeqPair0 stroke-width: 6px
    style Batch0 stroke-width: 6px
    subgraph drawbox.py
    Fig0
    end
    subgraph anime.py
    Fig0 -.- Anime0
    end
    subgraph raw.py
    Base0
    Raw0
    Raw1
    Raw2
    end
    subgraph flight.py
    Raw0 --> |Initialize| Flight0
    Raw1 --> |Initialize| Flight0
    Raw2 --> |Initialize| Flight0
    Flight0 --> |Initialize| Flight1
    end
    subgraph frame.py
    BasePair0
    Flight1 --> |Initialize| SeqPair0
    BasePair0 --> |Contain| Norm0
    SeqPair0 --> |Initialize| FramePair0
    end
    subgraph batch.py
    FramePair0 --> |Initialize| Batch0
    Batch0 --> |Contain| Iter0
    SeqPair0 --> |Initialize| Batch1
    FramePair0 --> |Initialize| Batch1
    Batch1 --> |Contain| Iter1
    end
    subgraph nnet.py
    Manager0
    end
```

## Run

- Data Folder

```
data
	 g1000
	 	020918
	 		Flight 1.xlsx
	 		Flight 2.csv
	 		...
	 	...
	 galaxy
	 	021918
	 		<file>
	 	...
	 pixel
	 	021918
	 		<file>
	 	...
	 stratux
	 	2018_02_09-07_45_27.csv
	 	...
```


- Training and Prediction

```bash
python main.py data output --phone galaxy --update --eval
python main.py output output --phone galaxy --batch-size 16 --model FC3 --win 32 --rate 0.4 --keyword pitch --trig --epochs 100 --lr 0.01 --print-freq 5 --cuda --device 0 --try 1
python main.py output output --phone stratux --batch-size 16 --model LSTM3 --win 32 --rate 0.4 --keyword roll --trig --stratux 1 --epochs 100 --lr 0.01 --print-freq 5 --cuda --device 0 --try 1
python main.py output output --phone galaxy --batch-size 16 --model FC3 --win 1 --offset 1 --keyword hazard --threshold 25 --direct --no-normal --epochs 100 --lr 0.01 --print-freq 5 --cuda --device 0 --try 10
```

- Animation

```bash
python anime.py data galaxy 032618 --start 1000
python test.py output --phone galaxy --model ./model/galaxy
```

If the selected date has several valid flights, it will require you to type flight ID in console (highlight with blue color).

## Risk

- Raw data time zone remains to be a hazard for possible new g1000, phone and stratux data

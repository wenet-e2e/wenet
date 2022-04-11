# wenet-go
go wrapper for wenet runtime
- non streamming example
```go
model, _ := wenet.Load("/root/model")
wav, _ := ioutil.ReadFile("/root/model/test.wav")

model.Recognize(wav[44:])
```

- streamming example
```go
# go run example.go
func main(){
    model, err := wenet.Load("/root/model")
    wav, err := ioutil.ReadFile("/root/model/test.wav")
    if err != nil{
        panic(err)
    }

    streamming := wenet.NewStreammingAsrDecoder(model, 1, false)
    go func(){
        nbytes := ((16000/1000)*2*200)
        for i := 0; i < len(wav); {
            r := i+nbytes
            final := false
            if r > len(wav){
                r = len(wav)
                final = true
            }
            streamming.AcceptWaveform(wav[i:r], final)
            i = r
        }
    }()
    for res := range streamming.Result{
        fmt.Println(res)
    }

}
```
```bash
[{"sentence":"甚至"}]
[{"sentence":"甚至出现"}]
[{"sentence":"甚至出现交易几"}]
[{"sentence":"甚至出现交易几乎停滞"}]
[{"sentence":"甚至出现交易几乎停滞的情"}]
[{"sentence":"甚至出现交易几乎停滞的情况","word_pieces":[{"word":"甚","start":0,"end":880},{"word":"至","start":880,"end":1120},{"word":"出","start":1120,"end":1400},{"word":"现","start":1400,"end":1720},{"word":"交","start":1720,"end":1960},{"word":"易","start":1960,"end":2160},{"word":"几","start":2160,"end":2400},{"word":"乎","start":2400,"end":2640},{"word":"停","start":2640,"end":2800},{"word":"滞","start":2800,"end":3040},{"word":"的","start":3040,"end":3240},{"word":"情","start":3240,"end":3600},{"word":"况","start":3600,"end":4160}]}]

```
- label check
```go

model, err := wenet.Load("/root/model")
_ = err_

labels := []string{"甚", "至", "出", "现", "交", "易", "几", "乎", "停", "滞", "的", "情", "好"}
checker := wenet.NewLabelChecker(model)

wav, _ := ioutil.ReadFile("/root/model/test.wav")
checker.Check(wav, labels)
```
```bash
[{"sentence":"甚至出现交易几乎停滞的情<is>况</is>","word_pieces":[{"word":"甚","start":0,"end":880},{"word":"至","start":880,"end":1120},{"word":"出","start":1120,"end":1400},{"word":"现","start":1400,"end":1720},{"word":"交","start":1720,"end":1960},{"word":"易","start":1960,"end":2120},{"word":"几","start":2120,"end":2400},{"word":"乎","start":2400,"end":2640},{"word":"停","start":2640,"end":2800},{"word":"滞","start":2800,"end":3040},{"word":"的","start":3040,"end":3240},{"word":"情","start":3240,"end":3600},{"word":"况","start":3600,"end":4160}]}]
```

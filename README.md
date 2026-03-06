# sieve-mask-spectral-peaks
## (연구 노트) Sieve-mask spectral peaks — “리만가설 아님” 공표용

이 섹션은 **리만가설(RH) 증명/반증**이 아니라, 우리가 관측한 “반복 주파수 피크”가 어디서 생기는지에 대한 **아티팩트(생성 메커니즘) 규명**이다.

- **핵심 요지**: 반복 피크는 “소수의 보편 상수”라기보다, **작은 소수 체가 만드는 0/1 마스크(문양) 주기열의 DFT(FFT) 공명 모드**로 설명된다.
- **강한 피크의 조건**: 강함은 0/1의 **개수**가 아니라 **배열/경계(질감) 스케일**에 의존하며, 로컬 블록 셔플(blur)로 붕괴한다.

### 재현(권장 커맨드)

사전조건:

```powershell
python -m pip install numpy
```

1) sieve sweep(체 단계별 원인추적 + \(\varphi(P_n)\), 약분/안정화 포함):

```powershell
python .\tools\riemann_experiment.py --mode sieve_sweep --N 2000000 --M 3000000 --Ws 30 210 --windows hann --axes k --L 65536 --num-windows 4 --K 30 --f-min-ratio 0.01 --track-delta-bins 4 --perms 10 --shuffle-block 2048 --seed 42 --out .\riemann_sieve_sweep.md
```

2) mask FFT(마스크만으로 타겟 bin/topK 확인 + 에너지/랭크):

```powershell
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft.md
```

3) 10차(룬 교체/제거 + 경계 흐리기 blur):

```powershell
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --mask-ablate --mask-swap-last --mask-blur-block 1024 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft_10th.md
```

4) blur 스케일 스윕(임계 경계 스케일 요약표 포함):

```powershell
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 10 --mask-blur-sweep 32 64 128 256 512 1024 2048 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft_blur_sweep.md
```

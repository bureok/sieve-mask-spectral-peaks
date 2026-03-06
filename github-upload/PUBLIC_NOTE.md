# PUBLIC NOTE (draft) — sieve-mask spectral peaks artifact

## TL;DR
이 글은 **리만가설(RH) 증명/반증**이 아니다.  
내가 소수열 기반 실험에서 반복적으로 관측한 “고주파 피크(특정 주파수들이 반복됨)”가, 소수의 보편 상수라기보다 **작은 소수 체(sieve)가 만드는 0/1 마스크 주기열의 DFT(FFT) 공명 모드**로 설명된다는 “아티팩트(생성 메커니즘) 규명”이다.

## 관측 현상
- 어떤 실험 파이프라인(후보열→정규화/잔차화→스펙트럼)에서 반복적으로 특정 \(f_\*\) 근방 피크가 나타남.
- 위상 랜덤화(PSD 유지)에서도 유지 → 고차 구조가 아니라 **2차 통계(PSD/ACF)** 기반 신호.

## 핵심 결과
1) **(원인) 체 마스크 주기열 → DFT bin**
- 소수성(primality)을 제거하고, “wheel + sieve primes”로 만든 **0/1 마스크**만으로 한 주기 시퀀스를 구성한 뒤 DFT를 계산하면,
  타겟 \(f_\*\)는 \(h^\*=\mathrm{round}(f_\*P)\)로 매핑되는 **bin 자체가 상위 피크(topK)**에 들어간다.

2) **(강함) 개수가 아니라 배열/경계(질감)**
- 0/1의 **개수는 유지**한 채, 로컬 블록 내부 셔플(경계 흐리기, blur)을 적용하면 강한 피크가 급격히 붕괴한다.
- `{7,11}` 단계에선 blur 블록 크기 **32~64**가 특히 치명적인 스케일로 관측됨(타겟별 최소 ratio).

## 왜 의미가 있나
이 결과는 RH를 푸는 게 아니라, 계산/실험에서 RH나 영점 신호를 찾는 과정에서 흔히 섞일 수 있는
**“sieve-mask 스펙트럼 아티팩트”**를 구분/제거하는 체크포인트를 제공한다.

## 재현 방법(Windows PowerShell)
사전조건:

```powershell
python -m pip install numpy
```

실행(결과는 md로 저장됨):

```powershell
python .\tools\riemann_experiment.py --mode sieve_sweep --N 2000000 --M 3000000 --Ws 30 210 --windows hann --axes k --L 65536 --num-windows 4 --K 30 --f-min-ratio 0.01 --track-delta-bins 4 --perms 10 --shuffle-block 2048 --seed 42 --out .\riemann_sieve_sweep.md
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft.md
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 25 --mask-ablate --mask-swap-last --mask-blur-block 1024 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft_10th.md
python .\tools\riemann_experiment.py --mode mask_fft --Ws 30 210 --sieve-primes 7 11 13 17 19 23 29 --sieve-steps 4 --mask-max-pn 600000 --mask-topk 10 --mask-blur-sweep 32 64 128 256 512 1024 2048 --f-min-ratio 0.01 --seed 42 --out .\riemann_mask_fft_blur_sweep.md
```

## 주의
- 이 결과는 **RH 증명이 아니다**. “관측된 반복 피크”가 어디서 생겼는지(필터/마스크의 주기 공명)를 밝힌 것이다.
- 동일 파이프라인을 쓰는 다른 실험에서도, 해석 전에 “sieve-mask 공명 여부”를 먼저 체크하는 걸 권장한다.

## 대표 타겟 \(f\) (이 실험에서 반복 관측된 값)
- `0.397728`
- `0.488640`
- `0.432693`
- `0.410713`

## 개인 메모(느낌)
“그것은 끈질기게 진실을 흐리게 만들었던 무언가의 연막 같았다. 하지만 이제는 선명해졌다.”
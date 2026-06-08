# Transformer

:::{admonition} 논문 정보
:class: note

- **제목**: Attention Is All You Need
- **저자**: Vaswani et al.
- **기관**: Google
- **발표**: NeurIPS 2017
- **arXiv**: [링크](https://arxiv.org/abs/1706.03762)
:::

## 핵심 아이디어

Transformer는 **Attention 연산만 사용**하여 Sequence Modeling을 한다는 접근을 제시한다. 

기존의 RNN, LSTM, GRU 아키텍처는 순차 연산 구조로 인해 
* (i) 멀리 떨어진 위치 간의 dependency 정보가 step을 거치며 점차 희석되고, 
* (ii) train parallelization이 어렵다는 문제를 갖는다.

이러한 문제점을 Transformer 아키텍처는 Attention을 사용해 효과적으로 해결한다. 

구체적으로, **Attention은 거리가 먼 위치 간의 dependency를 위치 거리와 무관하게 모델링이 가능하며, 구조적으로 병렬 연산이 가능**하다. 
특히 self-attention은 한 번의 layer에서 모든 token 쌍이 직접 연결되므로, RNN처럼 O(n) step을 거치지 않고 dependency를 모델링할 수 있다. 

## 방법론

### Transformer Model Architecture 

Transformer는 attention module로 조합된 **encoder-decoder 구조**이다. 
Encoder는 입력 문장을 이해하고, Decoder는 출력 문장을 생성한다. 

:::{image} ./images/transformer-architecture.png
:alt: alt text
:width: 350px
:align: center
:::

원 논문(Base Model)에서 **Encoder**는 N=6개의 동일한 레이어로 구성되며, 각 레이어는 두 개의 sub-layer로 이루어진다.

- **Multi-Head Self-Attention**: 입력 시퀀스의 모든 위치 쌍 간의 attention을 병렬로 계산한다.
- **Position-wise Feed-Forward Network(FFN)**: 각 위치에 독립적으로 적용되는 2층 fully connected network이다.

각 sub-layer를 지난 후에는 residual connection과 layer normalization(Post-LN)을 거친다. 

> **참고 — Pre-LN vs Post-LN**: 원 논문은 sub-layer 이후에 normalization을 적용하는 Post-LN 방식을 사용한다. 
반면 GPT-2 이후 대부분의 현대 모델은 sub-layer 이전에 normalization을 적용하는 Pre-LN을 채택하는데, 이는 학습 안정성이 더 높기 때문이다.

<br/>

**Decoder** 역시 N=6개의 동일한 레이어로 구성되며, 각 레이어는 세 개의 sub-layer로 이루어진다.

- **Masked Multi-Head Self-Attention**: output embedding을 입력으로 받아, masking을 통해 미래 위치를 보지 못하게 함으로써 auto-regressive하게 동작한다.
- **Multi-Head Cross-Attention**: encoder의 출력을 key/value로 사용해 encoder 정보를 decoder에 전달한다.
- **Position-wise FFN**: Encoder의 FFN과 동일한 구조이다.

Encoder와 마찬가지로 각 sub-layer 이후 residual connection과 Post-LN이 적용된다.

<br/>

최종적으로 Linear Layer와 Softmax를 통해 **각 시점의 target vocabulary에 대한 확률 분포**를 얻는다.

Machine Translation을 예로 들면, 학습 시에는 Teacher Forcing을 사용하여 이전 시점의 정답 단어를 Decoder 입력으로 제공하고, 모델이 다음 단어를 예측하도록 학습한다. 
반면 추론 시에는 모델이 예측한 토큰을 다시 Decoder의 입력으로 사용하며, 이를 \<EOS\> 토큰이 생성될 때까지 반복하여 번역 문장을 완성한다.

> **참고 — Teacher Forcing**: 학습 시 모델이 자신의 (잘못된) 예측을 다음 시점의 입력으로 사용하면 오류가 누적되어 학습이 불안정해질 수 있다. 이를 방지하기 위해 Teacher Forcing은 이전 시점의 예측값 대신 실제 정답(ground-truth) 토큰을 Decoder의 다음 입력으로 제공한다. 
다만 이 경우 학습 시와 추론 시 Decoder가 보는 입력의 분포가 달라지는 **exposure bias** 문제를 야기할 수 있다는 한계도 존재한다.

<!--
Machine Translation을 예로 들면, 입력 문장 "I am a student"를 "나는 학생이다"로 번역하는 경우를 생각해보자.

**학습 시 (Teacher Forcing)**

정답 시퀀스 `<BOS> 나는 학생이다 <EOS>`를 미리 알고 있으므로, 모든 시점의 입력을 한 번에 구성할 수 있다. t=1에서는 `<BOS>`를 입력으로 받아 `나는`을 예측하고, t=2에서는 `나는`을 입력으로 받아 `학생이다`를 예측하며, t=3에서는 `학생이다`를 입력으로 받아 `<EOS>`를 예측한다. 각 시점에 모델의 예측이 아닌 **정답 토큰**을 입력으로 제공하므로, 모든 시점을 병렬로 한 번에 처리할 수 있다.

**추론 시 (Auto-regressive)**

정답을 알 수 없으므로, 이전 시점에서 **모델이 예측한 토큰**을 다음 입력으로 사용하며 순차적으로 진행한다. t=1에서는 `<BOS>`만을 입력으로 받아 `나는`을 예측하고, t=2에서는 `<BOS> 나는`을 입력으로 받아 `학생이다`를 예측하며, t=3에서는 `<BOS> 나는 학생이다`를 입력으로 받아 `<EOS>`를 예측하면 생성을 종료한다. 이처럼 이전 예측이 확정되어야 다음 시점을 진행할 수 있으므로, **N개의 토큰을 생성하려면 Decoder를 N번 순차적으로 실행**해야 한다.

> **참고 — Teacher Forcing**: 학습 시 모델이 자신의 (잘못된) 예측을 다음 시점의 입력으로 사용하면 오류가 누적되어 학습이 불안정해질 수 있다. 이를 방지하기 위해 Teacher Forcing은 이전 시점의 예측값 대신 실제 정답(ground-truth) 토큰을 Decoder의 다음 입력으로 제공한다.
> 다만 이 경우 학습 시와 추론 시 Decoder가 보는 입력의 분포가 달라지는 **exposure bias** 문제를 야기할 수 있다는 한계도 존재한다.
--> 

<br/> 

### Attention

:::{image} ./images/transformer-attention.png
:alt: alt text
:width: 600px
:align: center
:::

Transformer가 사용하는 attention의 기본 형태는 **Scaled Dot-Product Attention**이다.
Query(Q), Key(K), Value(V) 세 벡터에 대해, Q와 K의 내적으로 유사도를 계산하고 이를 $\sqrt{d_k}$로 나누어 scaling한 뒤 softmax를 취해 attention weight를 얻고, 이 weight를 V에 가중합한다.

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$\sqrt{d_k}$로 scaling하는 이유는, $d_k$가 커질수록 dot-product 값의 분산이 커져 softmax 출력이 특정 위치에 지나치게 집중되고 gradient가 소실되는 문제를 방지하기 위함이다.

Transformer는 단일 attention 대신 **Multi-Head Attention**을 사용한다. Q, K, V를 $h$개의 서로 다른 linear projection으로 변환하여 각각 attention을 병렬로 계산한 뒤, 결과를 concat하고 다시 linear projection한다.

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
$$
$$
\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

이를 통해 모델은 서로 다른 representation subspace에서 다양한 관점의 정보를 동시에 포착할 수 있다. 원 논문(Base Model)에서는 $h=8$, $d_k = d_v = d_{\text{model}}/h = 64$를 사용한다.

<br/>

Transformer 아키텍처에서 attention은 세 가지 방식으로 사용된다.

- **Encoder Self-Attention**: encoder의 각 위치가 이전 layer의 모든 위치를 attend한다.
- **Decoder Masked Self-Attention**: decoder의 각 위치가 자기 자신을 포함한 이전 위치까지만 attend하며, masking으로 미래 위치에 대한 정보 유출을 차단한다.
- **Encoder-Decoder Cross-Attention**: decoder의 Query가 encoder 출력을 Key/Value로 사용해 attend하며, encoder가 인코딩한 입력 시퀀스 정보를 decoder로 전달한다.

<br/>

### Positional Encoding 

Transformer는 recurrence나 convolution을 전혀 사용하지 않으므로, 토큰의 순서(위치) 정보를 모델에 별도로 주입해주어야 한다. 이를 위해 input embedding에 Positional Encoding을 더하여 encoder와 decoder의 입력으로 사용한다.

논문에서는 서로 다른 주파수를 갖는 sine과 cosine 함수를 이용한 positional encoding을 사용한다. 짝수 차원에는 sin 함수를, 홀수 차원에는 cos 함수를 적용하며, 각 함수의 주기는 차원 인덱스에 따라 기하급수적으로 달라지도록 설계된다.

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right) \\
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

이러한 형태를 선택한 이유는, 임의의 고정된 offset $k$에 대해 위치 $(pos+k)$의 encoding이 위치 $pos$의 encoding에 대한 선형 변환으로 표현될 수 있어 모델이 상대적 위치(relative position) 관계를 학습하기 용이하고, 학습 시 접하지 못한 더 긴 시퀀스 길이에 대해서도 일반화할 수 있을 것으로 기대했기 때문이다. 

<br/>

## 실험 결과

### Machine Translation

:::{image} ./images/transformer-result-translation.png
:alt: alt text
:width: 500px
:align: center
:::

WMT 2014 English-to-German 번역 task에서 Transformer (big)는 **28.4 BLEU**를 기록하며, 기존 최고 성능 모델(앙상블 포함)을 2 BLEU 이상 앞서는 새로운 state-of-the-art를 달성했다. 
English-to-French task에서도 Transformer (big)는 **41.8 BLEU**로 단일 모델 기준 state-of-the-art를 기록했는데, 이때 학습 비용은 기존 최고 성능 모델의 1/4 수준에 불과했다. 
Base 모델 역시 기존에 발표된 모든 모델 및 앙상블 모델의 성능을 능가하면서도, 학습 비용은 훨씬 적게 소요되었다. 
이는 Transformer가 recurrence나 convolution 기반 아키텍처에 비해 **학습 속도와 병렬화 측면에서 크게 유리**하다는 점을 실험적으로 뒷받침한다.

> **참고 — BLEU Score**: 

<br/>

### Model Variation 

:::{image} ./images/transformer-result-modelvar.png
:alt: alt text
:width: 500px
:align: center
:::

Transformer 아키텍처의 각 구성 요소가 성능에 미치는 영향을 확인하기 위해 English-to-German 번역 task에서 ablation 실험을 진행했다.

- **Attention head 개수**: head 수를 1개로 줄이면 최적 설정 대비 BLEU가 0.9 하락했고, head 수를 지나치게 늘려도 오히려 성능이 떨어졌다. 즉 적절한 head 개수가 존재한다.
- **Attention key 차원($d_k$)**: $d_k$를 줄이면 모델 성능이 저하되었는데, 이는 dot-product 기반의 호환성(compatibility) 계산이 항상 최선은 아니며, 더 정교한 호환성 함수가 도움이 될 수 있음을 시사한다.
- **모델 크기**: 모델이 클수록(레이어 수, $d_{model}$ 등이 클수록) 성능이 향상되었다.
- **Dropout**: dropout을 적용하는 것이 overfitting 방지에 효과적이었다.
- **Positional Encoding**: sinusoidal 방식 대신 학습 가능한(learned) positional embedding을 사용해도 거의 동일한 성능을 보였다. 이는 두 방식이 실질적으로 비슷한 정보를 제공함을 보여주며, 그럼에도 sinusoidal 방식을 채택한 것은 학습 시 접하지 못한 더 긴 시퀀스에 대한 일반화 가능성 때문이다.

<br/>

## 한계점 

- **Self-attention의 연산 복잡도**: self-attention의 연산량과 메모리 사용량은 시퀀스 길이 $n$에 대해 $O(n^2 \cdot d)$로 증가한다. 따라서 시퀀스가 매우 길어지면 연산 비용이 급격히 커진다. 논문에서도 매우 긴 시퀀스를 다루기 위해서는 출력 위치 주변의 제한된 영역만 attend하도록 self-attention을 제한하는 방식을 향후 연구로 제시한다.
- **위치 정보의 후처리적 주입**: Transformer 구조 자체에는 순서 정보가 내재되어 있지 않으며, 이를 보완하기 위해 positional encoding을 별도로 더해주어야 한다. 즉 위치 정보 처리가 아키텍처에 본질적으로 내재되어 있다기보다는 추가적인 보완 장치에 가깝다.
- **추론 시 병렬화의 한계**: 학습 시에는 병렬 연산이 가능하지만, auto-regressive하게 단어를 하나씩 생성하는 추론(inference) 과정은 본질적으로 순차적이어서 Transformer의 병렬화 이점을 온전히 누리기 어렵다.
- **검증 범위의 한계**: 본 논문의 실험은 기계 번역(그리고 영어 구문 분석) task에 한정되어 있어, 다른 도메인이나 모달리티에 대한 일반화 가능성은 추후 연구를 통해 검증되어야 하는 과제로 남아 있다.

<br/>

## 인사이트

- **"Attention만으로 충분하다"는 패러다임 전환**: recurrence와 convolution 없이 attention만으로 sequence modeling이 가능함을 보임으로써, 이후 시퀀스 모델링 연구의 방향을 근본적으로 바꾸어 놓았다.
- **병렬화 가능한 구조가 가져온 scale-up**: 학습 과정의 병렬화가 가능해짐에 따라 더 큰 모델과 더 많은 데이터로 학습하는 것이 현실적으로 가능해졌고, 이는 이후 BERT, GPT 등 대규모 언어모델 시대로 이어지는 토대가 되었다.
- **거리에 무관한 dependency 모델링**: self-attention은 임의의 두 위치 사이를 단 한 번의 연산으로 직접 연결하므로(constant path length), 멀리 떨어진 위치 간의 dependency도 동일한 비용으로 모델링할 수 있다. 이는 이후 다양한 아키텍처 설계에 영향을 준 핵심적인 구조적 통찰이다.
- **범용적인 아키텍처 템플릿으로의 확장**: encoder-decoder 구조는 이후 task 특성에 따라 encoder만 사용하는 구조(BERT)나 decoder만 사용하는 구조(GPT)로 단순화·발전했으며, attention 메커니즘 자체도 vision, speech, multi-modal 등 NLP를 넘어선 영역으로 폭넓게 확장되었다.

<br/>

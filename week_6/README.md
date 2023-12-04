# LSTM

LSTM was introduced as an efficient gradient-based method to store information over extended time intervals. Traditional recurrent backpropagation methods like Back-Propagation Through Time (BPTT) and Real-Time Recurrent Learning (RTRL) were inadequate due to the exponential decay of error signals. LSTM addresses these issues by truncating the gradient in a way that doesn't harm the model and by using "constant error carousels" within special units to maintain a constant error flow

## Problem with Existing RNNs

Recurrent networks theoretically can store short-term memory but struggle with long minimal time lags between inputs and corresponding teacher signals. Traditional algorithms (BPTT, RTRL) suffer from either exploding or vanishing error signals, making them inefficient for learning over long time intervals​

## Design and Mechanism

LSTM is specifically designed to overcome error back-flow problems, enabling it to learn over time intervals exceeding 1000 steps, even with noisy input sequences​​.

LSTM compared with other methods such as Gradient-Descent Variants, Time-Delay Neural Networks, and Adaptive Sequence Chunkers, is superior when handling long time lag tasks​​. The use of multiplicative units (MUs) in LSTM differentiates it from other models, making it more efficient and suitable for long time lag problems​​.

A novel approach to constant error backpropagation was introduced, involving the use of memory cells and gate units to regulate the flow of errors and information​​. The internal state and outputs of the LSTM are computed through a set of equations that ensure constant error flow and efficient learning​

## Trade-offs

LSTM has the same update complexity per time step as BPTT but is local in space and time, making it more efficient and less memory-intensive​​.

LSTM has some problems such as the abuse of memory cells, allowing the network to find ways to reduce error without properly storing information over time, leading to memory cells being used as 'bias cells' and causing constant activation. 
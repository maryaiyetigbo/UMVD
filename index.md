<center>
<h1 style="display: block;">Unsupervised Video Denoising</h1>
CVMIA 2023 <br>
<table style="border: none; display: initial;">
<tr style="border: none;">
<td style="border: none;"><a href="https://maryaiyetigbo.github.io/">Mary Damilola Aiyetigbo</a><sup>1</sup></td>
<td style="border: none;"><a href="korte@clemson.edu">Alexander Korte</a><sup>1</sup></td>
<td style="border: none;"><a href="ema8@clemson.edu">Ethan Anderson</a><sup>1</sup></td>
<td style="border: none;"><a href="chalhoub@musc.edu">Reda Chalhoub</a><sup>2</sup></td>
<td style="border: none;"><a href="kalivasp@musc.edu">CPeter Kalivas</a><sup>2</sup></td>
<td style="border: none;"><a href="luofeng@clemson.edu">Feng Luo</a><sup>1</sup></td>
<td style="border: none;"><a href="nianyil@clemson.edu">Nianyi Li</a><sup>1</sup></td>
</tr>
</table>
<br>
<table style="border: none; display: initial;">
<tr style="border: none;">
<td style="border: none;"><sup>1</sup>Clemson University</td>
<td style="border: none;"><sup>2</sup>MUSC</td>
</tr>
</table>

</center>


## Two Photon Calcium Imaging
 <img src="./assets/highActivityb.gif" width="1000"/>

# Abstract

In this paper, we introduce a novel unsupervised network to denoise microscopy videos featured by image sequences captured by a fixed location microscopy camera. Specifically, we propose a DeepTemporal Interpolation method, leveraging a temporal signal filter integrated into the bottom CNN layers, to restore microscopy videos corrupted by unknown noise types. Our unsupervised denoising architecture is distinguished by its ability to adapt to multiple noise conditions without the need for pre-existing noise distribution knowledge, addressing a significant challenge in real-world medical applications. Furthermore, we evaluate our denoising framework using both real microscopy recordings and simulated data, validating our outperforming video denoising performance across a broad spectrum of noise scenarios. Extensive experiments demonstrate that our unsupervised model consistently outperforms state-of-the-art supervised and unsupervised video denoising techniques, proving especially effective for microscopy videos.


## Architecture
<img src="./assets/pipeline_fig.png" width="1000"/>

Our method consists of two main components: a feature generator $\mathcal{G}_\phi$ and a Denoiser $\mathcal{D}_\theta$. Distinct from conventional unsupervised methods, which directly use adjacent frames to interpolate the missing central frame or utilizing neighboring pixels for central pixel estimation as seen in UDVD \cite{udvd} and RDRF\cite{wang2023recurrent}, our technique employs a temporal filter on the feature maps produced by a sequence of CNN layers $\mathcal{G}_\phi$ prior to processing by the Denoiser $\mathcal{D}_\theta$. 
Our overall pipeline is illustrated in Fig.~\ref{fig:pipeline}. The major contributions of our work include:

Our DeepTemporal Interpolation Pipeline. The feature generator $\mathbf{F}_{\phi}$ extracts distinctive features using three depthwise convolutional layers, and a temporal filter enhances denoising accuracy by adjusting feature map weights based on spatial-temporal proximity to the central frame, overcoming challenges in handling high frame rate videos and slow-moving objects

Our goal is to generate denoised video $\{\hat{\mathbf{V}}_t|t = 1,2,...T\}$ given a sequence of noisy video input, $\{\mathbf{V}_t | t = 1,2,...T\}$, where $T$ is the total number of frames in the input video. In each iteration, a subset of contiguous frames $\{\mathbf{V}_t | t = 1,...c,...N\}$ is taken as input, with $N$ indicating the batch frame count and $c$ denoting the central frame's index. These frames are then passed through the feature generator $\mathbf{G}_{\phi}$, producing the corresponding feature maps $\{\mathbf{F}_t | t = 1,...,N\}$. Then, the temporal filter $\{ \gamma_t\}_{t=1}^N$ weights these feature maps $\{\mathbf{F}_t\}_{t=1}^N$, assigning diminished values to features nearer the central frame compared to those more distant. Next, the resulting weighted feature maps are concatenated and fed into the Denoiser $\mathbf{D}_{\theta}$, producing the denoised central frame $\hat{\mathbf{V}}_c$. A comprehensive visualization of our pipeline is provided in  Fig.~\ref{fig:pipeline}.

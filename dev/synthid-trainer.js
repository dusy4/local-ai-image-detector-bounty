const SIZE = 512, MAX_FREQ = 32, MIN_COHERENCE = 0.9, TOP_K = 64;
const status = document.querySelector("#status");
document.querySelector("#train").addEventListener("click", async () => {
  const positives = [...document.querySelector("#positive").files];
  const negatives = [...document.querySelector("#negative").files];
  if (positives.length < 8 || negatives.length < 8) return status.textContent = "Use at least 8 positive and 8 negative images; 50+ each is recommended.";
  status.textContent = "Extracting positive spectra…";
  const positiveSpectra = await spectra(positives);
  const candidates = selectCarriers(positiveSpectra);
  status.textContent = "Calibrating against negatives…";
  const negativeSpectra = await spectra(negatives);
  const positiveMatches = positiveSpectra.map(s => phaseMatch(s, candidates));
  const negativeMatches = negativeSpectra.map(s => phaseMatch(s, candidates));
  const positiveFloor = percentile(positiveMatches, 0.05), negativeCeiling = percentile(negativeMatches, 0.95);
  const center = (positiveFloor + negativeCeiling) / 2;
  const codebook = {
    version: "independent-phase-coherence-v1", imageSize: SIZE,
    sets: [{ name: "independently-derived", carriers: candidates.map(c => [c.fy, c.fx]), phases: candidates.map(c => c.phase) }],
    calibration: { center, steepness: 20, minimumReportScore: 0.65 },
    evidence: { positiveImages: positives.length, negativeImages: negatives.length, positiveFloor, negativeCeiling },
    note: "Derived locally from user-supplied references by the MIT-licensed Local Lens trainer."
  };
  status.textContent = JSON.stringify(codebook, null, 2);
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(codebook, null, 2)], { type: "application/json" })); a.download = "synthid-carriers.json"; a.textContent = "Download codebook"; status.after(a);
});

async function spectra(files) {
  const result = [];
  for (const [index, file] of files.entries()) {
    const bitmap = await createImageBitmap(file); const canvas = new OffscreenCanvas(SIZE, SIZE); const ctx = canvas.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(bitmap, 0, 0, SIZE, SIZE); bitmap.close(); const rgba = ctx.getImageData(0, 0, SIZE, SIZE).data;
    const re = new Float64Array(SIZE * SIZE), im = new Float64Array(SIZE * SIZE);
    for (let i = 0, p = 0; i < re.length; i++, p += 4) re[i] = (rgba[p] + rgba[p + 1] + rgba[p + 2]) / 3;
    fft2d(re, im); result.push({ re, im }); status.textContent = `FFT ${index + 1}/${files.length}`; await new Promise(r => setTimeout(r));
  }
  return result;
}

function selectCarriers(spectra) {
  const candidates = [];
  for (let fy = -MAX_FREQ; fy <= MAX_FREQ; fy++) for (let fx = -MAX_FREQ; fx <= MAX_FREQ; fx++) {
    if (Math.hypot(fy, fx) < 4) continue;
    const index = ((fy + SIZE) % SIZE) * SIZE + ((fx + SIZE) % SIZE); let unitRe = 0, unitIm = 0, magnitude = 0;
    for (const spectrum of spectra) { const phase = Math.atan2(spectrum.im[index], spectrum.re[index]); unitRe += Math.cos(phase); unitIm += Math.sin(phase); magnitude += Math.hypot(spectrum.re[index], spectrum.im[index]); }
    const coherence = Math.hypot(unitRe, unitIm) / spectra.length;
    if (coherence >= MIN_COHERENCE) candidates.push({ fy, fx, phase: Math.atan2(unitIm, unitRe), coherence, score: coherence * Math.log1p(magnitude / spectra.length) });
  }
  return candidates.sort((a, b) => b.score - a.score).slice(0, TOP_K);
}

function phaseMatch(spectrum, carriers) {
  return carriers.reduce((sum, c) => { const i = ((c.fy + SIZE) % SIZE) * SIZE + ((c.fx + SIZE) % SIZE); const phase = Math.atan2(spectrum.im[i], spectrum.re[i]); const d = Math.abs(Math.atan2(Math.sin(phase - c.phase), Math.cos(phase - c.phase))); return sum + 1 - d / Math.PI; }, 0) / carriers.length;
}
function percentile(values, q) { const sorted = [...values].sort((a,b)=>a-b); return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]; }

function fft2d(re, im) {
  const rowRe = new Float64Array(SIZE), rowIm = new Float64Array(SIZE);
  for (let y = 0; y < SIZE; y++) { const o = y * SIZE; for (let x = 0; x < SIZE; x++) { rowRe[x] = re[o+x]; rowIm[x] = im[o+x]; } fft(rowRe,rowIm); for (let x = 0; x < SIZE; x++) { re[o+x]=rowRe[x]; im[o+x]=rowIm[x]; } }
  for (let x = 0; x < SIZE; x++) { for (let y = 0; y < SIZE; y++) { rowRe[y]=re[y*SIZE+x]; rowIm[y]=im[y*SIZE+x]; } fft(rowRe,rowIm); for (let y = 0; y < SIZE; y++) { re[y*SIZE+x]=rowRe[y]; im[y*SIZE+x]=rowIm[y]; } }
}
function fft(re, im) {
  const n = re.length; for (let i=1,j=0;i<n;i++){ let bit=n>>1; for(;j&bit;bit>>=1)j^=bit; j^=bit; if(i<j){[re[i],re[j]]=[re[j],re[i]];[im[i],im[j]]=[im[j],im[i]];} }
  for(let len=2;len<=n;len<<=1){const angle=-2*Math.PI/len,wlenRe=Math.cos(angle),wlenIm=Math.sin(angle);for(let i=0;i<n;i+=len){let wr=1,wi=0;for(let j=0;j<len/2;j++){const u=i+j,v=i+j+len/2,vr=re[v]*wr-im[v]*wi,vi=re[v]*wi+im[v]*wr;re[v]=re[u]-vr;im[v]=im[u]-vi;re[u]+=vr;im[u]+=vi;const next=wr*wlenRe-wi*wlenIm;wi=wr*wlenIm+wi*wlenRe;wr=next;}}}
}

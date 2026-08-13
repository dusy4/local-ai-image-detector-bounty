import { env, pipeline } from "/public/vendor/transformers.min.js";
import { inspectMetadata } from "/dist/metadata.js";
import { fuseScores } from "/dist/shared.js";
env.allowRemoteModels=false; env.allowLocalModels=true; env.localModelPath="/public/models/"; env.backends.onnx.wasm.wasmPaths="/public/wasm/";
const samples=[{name:"Gemini attachment",url:"/samples/gemini-capybara.png"},{name:"OpenAI generated",url:"/samples/openai-capybara.png"}];
const codebook=await fetch("/research/private-synthid-carriers.json").then(r=>r.json());
const detector=await pipeline("image-classification","detector",{dtype:"q4",device:"wasm"});
const report=[];
for(const sample of samples){
  const response=await fetch(sample.url), bytes=new Uint8Array(await response.arrayBuffer()), blob=new Blob([bytes],{type:"image/png"});
  const result=await detector(sample.url,{topk:2});
  const fake=result.find(x=>/fake|deepfake|ai|generated/i.test(x.label));
  const real=result.find(x=>/real/i.test(x.label));
  const modelScore=fake?.score??(real?1-real.score:0.5);
  const watermark=await phaseScore(blob,codebook), metadataSignals=inspectMetadata(bytes), signals=[...metadataSignals];
  if(watermark.score>=0.65)signals.push({source:"SynthID experimental",score:watermark.score,detail:`${watermark.set} phase match ${watermark.phaseMatch}`});
  const finalAiScore=fuseScores(modelScore,signals);
  report.push({name:sample.name,modelAiScore:modelScore,metadataSignals,experimentalSynthId:watermark,finalAiScore,finalDecision:finalAiScore>=0.65?"AI":"real"});
}
document.querySelector("#result").textContent=JSON.stringify(report,null,2); document.title="DONE"; window.__report=report; await detector.dispose();

async function phaseScore(blob,book){
  const size=book.imageSize, bitmap=await createImageBitmap(blob), canvas=new OffscreenCanvas(size,size),ctx=canvas.getContext("2d",{willReadFrequently:true}); ctx.drawImage(bitmap,0,0,size,size); bitmap.close();
  const rgba=ctx.getImageData(0,0,size,size).data, pixels=new Float32Array(size*size); for(let i=0,p=0;i<pixels.length;i++,p+=4)pixels[i]=(rgba[p]+rgba[p+1]+rgba[p+2])/3;
  let best={set:null,phaseMatch:0,score:0};
  for(const set of book.sets){const phases=dftPhases(pixels,size,set.carriers),matches=phases.map((p,i)=>1-Math.abs(Math.atan2(Math.sin(p-set.phases[i]),Math.cos(p-set.phases[i])))/Math.PI),phaseMatch=matches.reduce((a,b)=>a+b,0)/matches.length,score=1/(1+Math.exp(-book.calibration.steepness*(phaseMatch-book.calibration.center)));if(score>best.score)best={set:set.name,phaseMatch,score};}
  return {...best,decision:best.score>=0.65?"watermark detected":"not detected",warning:"community codebook; experimental"};
}
function dftPhases(pixels,size,carriers){return carriers.map(([fy,fx])=>{let re=0,im=0;for(let y=0;y<size;y++)for(let x=0;x<size;x++){const a=-2*Math.PI*(fy*y+fx*x)/size,v=pixels[y*size+x];re+=v*Math.cos(a);im+=v*Math.sin(a);}return Math.atan2(im,re);});}

export type ModelInfo = { name:string; display_name:string; input_shape:number[]; is_loaded:boolean; description:string };
export type Prediction = { prediction_id:string; class_name:string; is_cancerous:boolean; probability:number; confidence:number; model_used:string; inference_time_ms:number; optimal_threshold:number; bounding_boxes:{x:number;y:number;width:number;height:number}[]; heatmap_base64?:string|null };
export type BatchItem = {slice_index:number;filename:string;class_name:string;is_cancerous:boolean;probability:number;confidence:number;anomaly_rank:number};
export type BatchResult = {batch_id:string;total_slices:number;cancerous_count:number;normal_count:number;highest_probability:number;top_suspicious_slice:string|null;slices:BatchItem[]};
const API = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000').replace(/\/$/, '');
async function request<T>(path:string, init?:RequestInit):Promise<T>{ const r=await fetch(`${API}${path}`,init); if(!r.ok){let msg=`Request failed (${r.status})`; try{const d=await r.json();msg=d.detail||msg}catch{} throw new Error(msg)} return r.json(); }
export const api={
  models:()=>request<{default_model:string;total_models:number;models:ModelInfo[]}>('/api/models'),
  health:()=>request<{status:string;version:string;loaded_models:string[];gpu_available:boolean;timestamp:string}>('/api/health'),
  predict:(file:File,model:string,gradcam=true)=>{const f=new FormData();f.append('file',file);f.append('model_name',model);f.append('generate_gradcam',String(gradcam));return request<Prediction>('/api/predict',{method:'POST',body:f})},
  batch:(files:File[],model:string)=>{const f=new FormData();files.forEach(x=>f.append('files',x));f.append('model_name',model);return request<BatchResult>('/api/predict/batch',{method:'POST',body:f})},
  reportUrl:(p:Prediction,patient='PATIENT-001')=>`${API}/api/report/download?${new URLSearchParams({patient_id:patient,class_name:p.class_name,probability:String(p.probability),confidence:String(p.confidence),model_name:p.model_used,optimal_threshold:String(p.optimal_threshold)})}`
};
export {API};

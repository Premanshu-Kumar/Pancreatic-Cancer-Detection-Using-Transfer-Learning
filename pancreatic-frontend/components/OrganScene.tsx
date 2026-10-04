'use client';
import {Canvas,useFrame} from '@react-three/fiber';
import {Environment,Float,OrbitControls,PerspectiveCamera,Stars} from '@react-three/drei';
import {useRef} from 'react';
import * as THREE from 'three';
function Organ(){const g=useRef<THREE.Group>(null);useFrame((_,d)=>{if(g.current)g.current.rotation.y+=d*.18});return <group ref={g} rotation={[.15,-.35,.08]}>
  <mesh scale={[2.5,1.05,.7]} position={[0,0,0]}><sphereGeometry args={[1,64,32]}/><meshPhysicalMaterial color="#d58b93" roughness={.48} metalness={.08} clearcoat={.35}/></mesh>
  <mesh scale={[1.4,.48,.42]} position={[2.05,-.08,0]} rotation={[0,0,-.18]}><sphereGeometry args={[1,48,24]}/><meshPhysicalMaterial color="#c87582" roughness={.5}/></mesh>
  <mesh scale={[.58,.58,.52]} position={[-2.05,.18,.04]}><sphereGeometry args={[1,48,24]}/><meshPhysicalMaterial color="#e2a0a4" roughness={.5}/></mesh>
  <mesh position={[.35,.12,.72]} rotation={[0,0,.25]}><torusGeometry args={[.45,.055,12,48]}/><meshBasicMaterial color="#69e3d1" transparent opacity={.75}/></mesh>
  <pointLight position={[.2,.4,2]} intensity={4} color="#72f3df" distance={6}/>
</group>}
export default function OrganScene(){return <div className="scene"><Canvas dpr={[1,2]}><PerspectiveCamera makeDefault position={[0,0,7]}/><ambientLight intensity={1.2}/><directionalLight position={[3,4,5]} intensity={3}/><Float speed={1.2} rotationIntensity={.15} floatIntensity={.25}><Organ/></Float><Stars radius={12} depth={5} count={500} factor={1.3} saturation={0} fade speed={.4}/><Environment preset="city"/><OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={.35}/></Canvas><div className="scan-ring"/><div className="scene-label">ANATOMICAL VISUALIZATION · NOT DIAGNOSTIC</div></div>}

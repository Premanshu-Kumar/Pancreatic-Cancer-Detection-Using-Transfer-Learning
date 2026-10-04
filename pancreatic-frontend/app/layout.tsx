import './globals.css';
import type { Metadata } from 'next';
export const metadata:Metadata={title:'PancreaAI — Clinical Imaging Intelligence',description:'Transfer-learning research interface for pancreatic CT analysis and explainable AI.'};
export default function RootLayout({children}:{children:React.ReactNode}){return <html lang="en"><body>{children}</body></html>}

import { useId } from 'react'
import type { CSSProperties } from 'react'

type ShapeProps = { className?: string; style?: CSSProperties; depth?: number }

function frame(className = '', style?: CSSProperties, depth?: number) {
  return {
    className: `shape ${className}`,
    style,
    'data-depth': depth,
    viewBox: '0 0 200 200',
    'aria-hidden': true,
  }
}

export function Star3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <radialGradient id={id} cx="32%" cy="26%" r="80%">
          <stop offset="0" stopColor="#F8E2FF" /><stop offset=".35" stopColor="#C77BFF" /><stop offset=".75" stopColor="#9A2EF2" /><stop offset="1" stopColor="#5A0DB0" />
        </radialGradient>
      </defs>
      <path d="M100 18 L124 74 L184 78 L138 117 L153 176 L100 144 L47 176 L62 117 L16 78 L76 74 Z" fill={`url(#${id})`} stroke={`url(#${id})`} strokeWidth="22" strokeLinejoin="round" />
      <ellipse cx="78" cy="70" rx="26" ry="12" fill="#fff" opacity=".55" transform="rotate(-28 78 70)" />
    </svg>
  )
}

export function Blob3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <radialGradient id={id} cx="35%" cy="28%" r="80%">
          <stop offset="0" stopColor="#F4FFF8" /><stop offset=".4" stopColor="#A7F0C7" /><stop offset=".85" stopColor="#4FBF86" /><stop offset="1" stopColor="#2E8F60" />
        </radialGradient>
      </defs>
      <ellipse cx="100" cy="100" rx="70" ry="88" fill={`url(#${id})`} transform="rotate(24 100 100)" />
      <ellipse cx="76" cy="62" rx="20" ry="10" fill="#fff" opacity=".6" transform="rotate(-30 76 62)" />
    </svg>
  )
}

export function Torus3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#FFE0A8" /><stop offset=".35" stopColor="#FFA23A" /><stop offset=".7" stopColor="#FF6A00" /><stop offset="1" stopColor="#B83C00" />
        </linearGradient>
      </defs>
      <circle cx="100" cy="100" r="62" fill="none" stroke={`url(#${id})`} strokeWidth="44" />
      <path d="M58 62 A62 62 0 0 1 118 41" fill="none" stroke="#fff" strokeWidth="9" strokeLinecap="round" opacity=".55" />
    </svg>
  )
}

export function Cross3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <radialGradient id={id} cx="35%" cy="28%" r="85%">
          <stop offset="0" stopColor="#E4FFCC" /><stop offset=".4" stopColor="#6BEB4A" /><stop offset=".85" stopColor="#22B11B" /><stop offset="1" stopColor="#0E6E0C" />
        </radialGradient>
      </defs>
      <path d="M78 20h44a10 10 0 0 1 10 10v38h38a10 10 0 0 1 10 10v44a10 10 0 0 1-10 10h-38v38a10 10 0 0 1-10 10H78a10 10 0 0 1-10-10v-38H30a10 10 0 0 1-10-10V78a10 10 0 0 1 10-10h38V30a10 10 0 0 1 10-10z" fill={`url(#${id})`} stroke={`url(#${id})`} strokeWidth="16" strokeLinejoin="round" />
      <rect x="82" y="30" width="12" height="40" rx="6" fill="#fff" opacity=".55" />
    </svg>
  )
}

export function Sphere3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <radialGradient id={id} cx="32%" cy="26%" r="80%">
          <stop offset="0" stopColor="#F8E2FF" /><stop offset=".35" stopColor="#C77BFF" /><stop offset=".75" stopColor="#9A2EF2" /><stop offset="1" stopColor="#5A0DB0" />
        </radialGradient>
      </defs>
      <circle cx="100" cy="100" r="80" fill={`url(#${id})`} />
      <ellipse cx="72" cy="66" rx="24" ry="13" fill="#fff" opacity=".6" transform="rotate(-30 72 66)" />
    </svg>
  )
}

export function Capsule3D({ className, style, depth }: ShapeProps) {
  const id = useId()
  return (
    <svg {...frame(className, style, depth)}>
      <defs>
        <radialGradient id={id} cx="35%" cy="28%" r="85%">
          <stop offset="0" stopColor="#E0FFFF" /><stop offset=".45" stopColor="#5EE0E6" /><stop offset="1" stopColor="#0D8C9A" />
        </radialGradient>
      </defs>
      <rect x="30" y="60" width="140" height="80" rx="40" fill={`url(#${id})`} transform="rotate(-18 100 100)" />
      <ellipse cx="72" cy="80" rx="26" ry="8" fill="#fff" opacity=".55" transform="rotate(-18 72 80)" />
    </svg>
  )
}

export function SunDoodle({ className = '', style }: ShapeProps) {
  return (
    <svg className={`shape ${className}`} style={style} viewBox="0 0 120 120" aria-hidden="true">
      <g fill="none" stroke="#F2CF1D" strokeWidth="4" strokeLinecap="round">
        <path d="M60 42c11-1 19 7 18 17s-9 18-19 17-17-8-16-18 7-15 17-16z" />
        <path d="M60 8v18M60 94v18M8 60h18M94 60h18M23 23l12 12M85 85l12 12M97 23L85 35M35 85L23 97" />
      </g>
    </svg>
  )
}

export function StarDoodle({ className = '', style }: ShapeProps) {
  return (
    <svg className={`shape ${className}`} style={style} viewBox="0 0 120 120" aria-hidden="true">
      <path d="M60 10c4 18 8 30 14 36 9 1 22 2 36 6-13 6-24 11-30 17 2 13 6 26 8 41-11-11-20-20-28-26-10 7-20 16-34 26 5-15 9-27 12-37-9-8-19-14-30-20 15-2 27-3 36-6 5-10 9-22 16-37z" fill="none" stroke="#F2CF1D" strokeWidth="3.5" strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  )
}

export function Sparks({ className = '', style }: ShapeProps) {
  return (
    <svg className={`shape ${className}`} style={style} viewBox="0 0 60 60" aria-hidden="true">
      <g stroke="#2F7BFF" strokeWidth="3.5" strokeLinecap="round"><path d="M14 30l6-14M32 8l-2 16M50 26l-14 6" /></g>
    </svg>
  )
}

export function Squiggle({ delay = 1 }: { delay?: number }) {
  return (
    <svg className="squiggle draw" style={{ '--draw-delay': `${delay}s` } as CSSProperties} viewBox="0 0 300 30" preserveAspectRatio="none" aria-hidden="true">
      <path pathLength={1} d="M4 18c18-14 30 10 48 0s30-14 48 0 30 10 48 0 30-14 48 0 30 10 48 0 30-12 48-2" fill="none" stroke="currentColor" strokeWidth="5" strokeLinecap="round" />
    </svg>
  )
}

export function Words({ text, start = 0, step = 75 }: { text: string; start?: number; step?: number }) {
  const words = text.split(' ')
  return (
    <>
      {words.map((word, index) => (
        <span key={`${word}-${index}`}>
          <span className="word-up" style={{ animationDelay: `${start + index * step}ms` }}>{word}</span>
          {index < words.length - 1 ? ' ' : ''}
        </span>
      ))}
    </>
  )
}

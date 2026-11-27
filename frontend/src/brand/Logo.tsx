import { useState } from 'react'

type LogoVariant = 'static' | 'once' | 'loop' | 'think'
type LogoTone = 'light' | 'dark'

const HOOK = 'M8 18a12 12 0 1 1 18 10c-4 2-6 5-6 10'
const BAR = 'M20 8v30'

export function LogoMark({ size = 32, variant = 'static', tone = 'light', title }: {
  size?: number
  variant?: LogoVariant
  tone?: LogoTone
  title?: string
}) {
  const ink = tone === 'dark' ? '#F6F5F1' : '#15161A'
  return (
    <svg width={Math.round(size * 40 / 74)} height={size} viewBox="0 -4 40 74" overflow="visible"
      className={`dd-mark dd-${variant} shrink-0`}
      role={title ? 'img' : undefined} aria-label={title} aria-hidden={title ? undefined : true}>
      <path className="dd-hook" pathLength={1} d={HOOK} fill="none" stroke={ink} strokeWidth="8" strokeLinecap="round" />
      <path className="dd-bar" d={BAR} fill="none" stroke={ink} strokeWidth="9" strokeLinecap="round" />
      <g className="dd-drop">
        <circle className="dd-dot" cx="20" cy="56" r="6.5" fill="#FFE45C" stroke={tone === 'dark' ? 'none' : ink} strokeWidth="2" />
      </g>
    </svg>
  )
}

export function Wordmark({ size = 32, variant = 'static', tone = 'light', textClassName = 'text-[22px]' }: {
  size?: number
  variant?: LogoVariant
  tone?: LogoTone
  textClassName?: string
}) {
  const [replay, setReplay] = useState(0)
  const drawn = variant === 'once'
  return (
    <span className={`${drawn ? '' : 'dd-hover '}flex items-center gap-2`} onMouseEnter={drawn ? () => setReplay(count => count + 1) : undefined}>
      <LogoMark key={replay} size={size} variant={variant} tone={tone} />
      <span className={`font-bold leading-none tracking-[-0.03em] ${tone === 'dark' ? 'text-white' : 'text-ink'} ${textClassName}`}>DoubtDesk</span>
    </span>
  )
}

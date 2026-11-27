import { useEffect, useRef, useState } from 'react'
import type { CSSProperties, FormEvent, ReactNode } from 'react'
import { ArrowRight, Check, LoaderCircle, LockKeyhole, Mail } from 'lucide-react'
import { authApi } from './api'
import type { AuthUser } from './api'
import { Wordmark } from '../brand/Logo'
import { Blob3D, Capsule3D, Cross3D, Sparks, Sphere3D, Squiggle, Star3D, StarDoodle, SunDoodle, Torus3D, Words } from '../brand/Shapes'

type Mode = 'login' | 'signup' | 'forgot' | 'verify' | 'reset' | 'check-email'

function initialRoute(): { mode: Mode; token: string } {
  const path = window.location.pathname.toLowerCase()
  const params = new URLSearchParams(window.location.search)
  const token = params.get('token') || ''
  if (path.endsWith('/verify-email')) return { mode: 'verify', token }
  if (path.endsWith('/reset-password')) return { mode: 'reset', token }
  return { mode: 'login', token: '' }
}

const COPYRIGHT_YEAR = new Date().getFullYear()

const PLANS = [
  { name: 'Free', price: '₹0', period: '', questions: '30', tokens: '200k', concurrent: '1', featured: false },
  { name: 'Pro', price: '₹199', period: '/ month', questions: '300', tokens: '3M', concurrent: '3', featured: true },
  { name: 'Unlimited', price: '₹799', period: '/ month', questions: 'No limit', tokens: 'No limit', concurrent: 'No limit', featured: false },
]

const LABELS = [
  { stamp: 'stamp--grounded', label: 'From documents', text: 'Taken from the PDFs you ticked, with page citations.' },
  { stamp: 'stamp--general', label: 'Not from document', text: 'General knowledge. Your PDFs didn’t cover it, or none were ticked.' },
  { stamp: 'stamp--failed', label: 'Verification unavailable', text: 'The answer couldn’t be checked against the source. Read with care.' },
  { stamp: 'stamp--scope', label: 'Outside academic scope', text: 'Not a study question, so it politely declines.' },
]

const STEPS = [
  { title: 'Upload a PDF', text: 'It’s read page by page, including diagrams. A document you’ve uploaded before is reused straight away.' },
  { title: 'Tick the documents, ask', text: 'Pick one PDF or several. Leave them all unticked to ask a general question instead.' },
  { title: 'Check the page', text: 'Answers are checked against your documents before you see them. Open any citation to read the source yourself.' },
]

const FEATURES = [
  { title: 'Every claim points to a page you can open.', text: 'Click a marked number and the PDF page opens beside the chat, at the passage it came from.' },
  { title: 'Checked before you see it.', text: 'Answers are verified against your documents, then revised if anything doesn’t hold up.' },
  { title: 'Maths renders properly.', text: 'Equations, tables and code come out formatted, not as raw symbols.' },
]

const STATS = [
  { value: '4', text: 'answer labels, so you know how far to trust it' },
  { value: '1 click', text: 'from a citation to the PDF page' },
  { value: '3', text: 'steps, no setup' },
  { value: '24h', text: 'rolling limit window' },
]

const BELIEF = 'We believe a tutor should stick to the facts, not make them up. Every answer points to the page it came from.'

const NAV = [
  { href: '#how', label: 'How it works', id: 'how' },
  { href: '#labels', label: 'Answer labels', id: 'labels' },
  { href: '#plans', label: 'Plans', id: 'plans' },
]

function prefersReducedMotion() {
  return typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches
}

function SectionTitle({ muted, strong, dark = false, center = false }: { muted?: string; strong: ReactNode; dark?: boolean; center?: boolean }) {
  return (
    <h2 className={`scroll-in text-balance text-[clamp(32px,4.4vw,54px)] font-semibold leading-[1.1] tracking-[-0.035em] ${center ? 'text-center' : ''}`}>
      {muted && <span className={dark ? 'text-[#5A5A63]' : 'text-ghost'}>{muted} </span>}{strong}
    </h2>
  )
}

export function AuthScreen({ onLogin }: { onLogin: (user: AuthUser) => void }) {
  const initial = useRef(initialRoute())
  const rootRef = useRef<HTMLDivElement>(null)
  const beliefRef = useRef<HTMLHeadingElement>(null)
  const [mode, setMode] = useState<Mode>(initial.current.mode)
  const [token, setToken] = useState(initial.current.token)
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [identifier, setIdentifier] = useState('')
  const [password, setPassword] = useState('')
  const [confirmPassword, setConfirmPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [beliefShown, setBeliefShown] = useState(false)
  const [feature, setFeature] = useState(0)
  const [activeSection, setActiveSection] = useState('')

  useEffect(() => {
    if (initial.current.token) {
      window.history.replaceState({}, '', '/')
    }
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => setFeature(current => (current + 1) % FEATURES.length), 4500)
    return () => window.clearInterval(timer)
  }, [])

  useEffect(() => {
    const target = beliefRef.current
    if (!target || typeof IntersectionObserver === 'undefined') {
      setBeliefShown(true)
      return
    }
    const observer = new IntersectionObserver(entries => {
      if (entries.some(entry => entry.isIntersecting)) {
        setBeliefShown(true)
        observer.disconnect()
      }
    }, { threshold: 0.4 })
    observer.observe(target)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (typeof IntersectionObserver === 'undefined') return
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => { if (entry.isIntersecting) setActiveSection(entry.target.id) })
    }, { rootMargin: '-45% 0px -45% 0px' })
    NAV.forEach(item => {
      const section = document.getElementById(item.id)
      if (section) observer.observe(section)
    })
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (prefersReducedMotion()) return
    function onMove(event: PointerEvent) {
      if (event.pointerType !== 'mouse' || !rootRef.current) return
      const x = event.clientX / window.innerWidth - 0.5
      const y = event.clientY / window.innerHeight - 0.5
      rootRef.current.querySelectorAll<SVGElement>('[data-depth]').forEach(shape => {
        const depth = Number(shape.dataset.depth)
        shape.style.translate = `${(-x * depth).toFixed(1)}px ${(-y * depth).toFixed(1)}px`
      })
    }
    window.addEventListener('pointermove', onMove)
    return () => window.removeEventListener('pointermove', onMove)
  }, [])

  function changeMode(next: Mode) {
    setMode(next)
    setError('')
    setNotice('')
    setPassword('')
    setConfirmPassword('')
    if (next === 'login' && identifier === '' && email) setIdentifier(email)
  }

  function openAccount(next: Mode) {
    if (mode !== next) changeMode(next)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (busy) return
    setError('')
    setNotice('')
    if ((mode === 'signup' || mode === 'reset') && password !== confirmPassword) {
      setError('Passwords do not match.')
      return
    }
    setBusy(true)
    try {
      if (mode === 'login') {
        const user = await authApi.login(identifier.trim(), password)
        onLogin(user)
      } else if (mode === 'signup') {
        await authApi.signup(username.trim(), email.trim(), password)
        changeMode('check-email')
        setNotice('Account created. Check your inbox for a verification link before logging in.')
      } else if (mode === 'forgot') {
        const result = await authApi.forgot(email.trim())
        setNotice(result.message)
      } else if (mode === 'verify') {
        const result = await authApi.verify(token)
        setToken('')
        changeMode('login')
        setNotice(result.message + ' You can now sign in.')
      } else if (mode === 'reset') {
        const result = await authApi.reset(token, password)
        setToken('')
        changeMode('login')
        setNotice(result.message + ' Sign in with your new password.')
      } else if (mode === 'check-email') {
        const result = await authApi.resend(email.trim())
        setNotice(result.message)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Please try again.')
    } finally {
      setBusy(false)
    }
  }

  const titles: Record<Mode, string> = {
    login: 'Welcome back',
    signup: 'Create your account',
    forgot: 'Reset your password',
    verify: 'Verify your email',
    reset: 'Choose a new password',
    'check-email': 'Check your inbox',
  }
  const descriptions: Record<Mode, string> = {
    login: 'Sign in to access your PDFs and learning workspace.',
    signup: 'Your documents and conversations stay in your own account.',
    forgot: 'We’ll email you a password-reset link if your account is eligible.',
    verify: 'Confirm your email address to activate your account.',
    reset: 'Enter a new password to finish your reset.',
    'check-email': 'Verification links expire after 10 minutes.',
  }

  const labelClass = 'rise flex flex-col gap-1.5 text-[13px] font-medium text-[#111114]'
  const linkClass = 'underline decoration-[#D5D5D2] underline-offset-4 transition hover:text-[#111114] hover:decoration-[#111114]'
  const showTabs = mode === 'login' || mode === 'signup'
  const beliefWords = BELIEF.split(' ')

  return (
    <div ref={rootRef} className="min-h-dvh overflow-x-hidden bg-paper text-ink">
      <main id="top">
        <section className="relative overflow-hidden rounded-b-[30px] bg-night pb-24 pt-7 text-white sm:rounded-b-[44px]">
          <Star3D depth={28} className="float -right-10 top-[230px] w-[110px] sm:top-[70px] sm:w-[230px]" style={{ '--r': '-10deg' } as CSSProperties} />
          <Blob3D depth={18} className="float float-2 right-[200px] top-5 hidden w-[150px] sm:block" />
          <Torus3D depth={34} className="float float-3 -bottom-[70px] -left-[90px] w-[260px]" style={{ '--r': '20deg' } as CSSProperties} />
          <Sphere3D depth={12} className="float bottom-8 left-[44%] w-20" />
          <div className="relative z-[2] mx-auto max-w-[1200px] px-4 sm:px-6">
            <header className="flex items-center justify-between gap-4">
              <a href="#top" aria-label="DoubtDesk home" className="text-white no-underline"><Wordmark size={40} variant="once" tone="dark" /></a>
              <a href="#account" onClick={() => openAccount('login')} className="rounded-full border border-[#33333B] px-[18px] py-[9px] text-sm font-medium transition hover:bg-night-2">Sign in</a>
            </header>

            <div className="mt-16 flex flex-wrap items-center gap-14">
              <div className="min-w-0 flex-[999_1_520px]">
                <p className="pill pill--lime rise"><span className="pill-dot" />STEM doubt tutor · Biology, Chemistry, Physics, Maths</p>
                <h1 className="mt-6 text-balance text-[clamp(42px,6.6vw,84px)] font-semibold leading-[1.04] tracking-[-0.035em]">
                  <Words text="Stuck on a STEM doubt?" start={150} /><br />
                  <Words text="Get it from" start={450} />{' '}<span className="relative inline-block"><Words text="your own book." start={675} /><Squiggle delay={1.2} /></span>
                </h1>
                <p className="rise mt-7 max-w-[520px] text-lg font-light leading-[1.7] text-[#B4B4BD] [animation-delay:.7s]">
                  Upload your Biology, Chemistry, Physics or Maths PDFs and ask your doubt. DoubtDesk answers from your pages, opens the exact page it used, and says so when an answer comes from general knowledge instead. Study questions only.
                </p>
                <dl className="mt-10 flex flex-wrap gap-3">
                  {[['PDFs', 'Up to 25 MB each'], ['History', '30 saved chats'], ['Maths', 'Equations render']].map(([term, value], index) => (
                    <div key={term} className="rise rounded-full border border-white/10 bg-white/[.06] px-[22px] py-3.5 transition hover:-translate-y-1 hover:bg-white/10" style={{ animationDelay: `${900 + index * 100}ms` }}>
                      <dt className="text-xs text-night-muted">{term}</dt><dd className="mt-0.5 font-semibold">{value}</dd>
                    </div>
                  ))}
                </dl>
              </div>

              <section id="account" aria-label="Account" className="slide-in w-full min-w-0 max-w-[440px] flex-[1_1_360px] scroll-mt-6 rounded-[26px] bg-white p-6 text-[#111114] shadow-[0_30px_80px_rgba(0,0,0,.45)] sm:rounded-[32px] sm:p-[34px]">
                <h2 className="text-[30px] font-semibold leading-tight tracking-[-0.025em]">{titles[mode]}</h2>
                <p className="mt-1.5 text-[14.5px] leading-6 text-[#6E6E78]">{descriptions[mode]}</p>

                {showTabs && <div role="tablist" aria-label="Account mode" className="relative mt-5 flex rounded-full bg-[#F1F1EF] p-1">
                  <span aria-hidden="true" className={`absolute bottom-1 left-1 top-1 w-[calc(50%-4px)] rounded-full bg-night transition-transform duration-[350ms] ease-[cubic-bezier(.3,1.3,.5,1)] ${mode === 'signup' ? 'translate-x-full' : ''}`} />
                  {(['login', 'signup'] as const).map(tab => (
                    <button key={tab} type="button" role="tab" aria-selected={mode === tab} onClick={() => openAccount(tab)}
                      className={`relative z-[1] flex-1 rounded-full py-2.5 text-sm font-medium transition-colors duration-300 ${mode === tab ? 'text-white' : 'text-[#6E6E78] hover:text-[#111114]'}`}>
                      {tab === 'login' ? 'Sign in' : 'Create account'}
                    </button>
                  ))}
                </div>}

                <form onSubmit={(event) => void submit(event)} className="mt-6 space-y-4">
                  {mode === 'signup' && <label className={labelClass}>Username
                    <input className="field w-full font-normal" aria-label="Username" autoComplete="username" required minLength={3} maxLength={32} pattern="[A-Za-z0-9_]+" value={username} onChange={(event) => setUsername(event.target.value)} placeholder="your_username" />
                  </label>}
                  {(mode === 'signup' || mode === 'forgot' || mode === 'check-email') && <label className={labelClass}>Email
                    <input className="field w-full font-normal" type="email" aria-label="Email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" />
                  </label>}
                  {mode === 'login' && <label className={labelClass}>Username or email
                    <input className="field w-full font-normal" aria-label="Username or email" autoComplete="username" required value={identifier} onChange={(event) => setIdentifier(event.target.value)} placeholder="Username or email" />
                  </label>}
                  {(mode === 'signup' || mode === 'login' || mode === 'reset') && <label className={labelClass}>{mode === 'reset' ? 'New password' : 'Password'}
                    <input className="field w-full font-normal" type="password" aria-label={mode === 'reset' ? 'New password' : 'Password'} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={mode === 'login' ? 1 : 8} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} placeholder={mode === 'login' ? 'Enter your password' : 'At least 8 characters'} />
                  </label>}
                  {(mode === 'signup' || mode === 'reset') && <label className={labelClass}>Confirm password
                    <input className="field w-full font-normal" type="password" aria-label="Confirm password" autoComplete="new-password" required minLength={8} maxLength={128} value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} placeholder="Repeat your password" />
                  </label>}
                  {mode === 'verify' && <div className="flex items-center gap-2 rounded-[18px] bg-[#DDF8E9] p-3 text-sm text-[#111114]"><Mail size={18} />{token ? 'Verification link received.' : 'No verification token was found in this link.'}</div>}
                  {mode === 'reset' && !token && <p role="alert" className="text-sm text-[#D93036]">No reset token was found. Request a new link.</p>}
                  {notice && <p role="status" className="rise rounded-[18px] bg-[#DDF8E9] px-3.5 py-2.5 text-sm leading-6 text-[#111114]">{notice}</p>}
                  {error && <p role="alert" className="rise rounded-[18px] bg-[#FDECEC] px-3.5 py-2.5 text-sm leading-6 text-[#D93036]">{error}</p>}
                  <button className="primary-button !mt-6 h-[54px] w-full justify-center text-[15.5px]" type="submit" disabled={busy || ((mode === 'verify' || mode === 'reset') && !token)}>
                    {busy && <LoaderCircle className="animate-spin" size={17} />}
                    {{ login: 'Sign in', signup: 'Create account', forgot: 'Send reset link', verify: 'Verify email', reset: 'Reset password', 'check-email': 'Resend verification email' }[mode]}
                    {!busy && <span className="arrow-dot" aria-hidden="true"><ArrowRight size={15} /></span>}
                  </button>
                </form>
                <div className="mt-6 flex flex-col items-start gap-2.5 border-t border-[#EEEEEC] pt-5 text-[13.5px] text-[#6E6E78]">
                  {mode === 'login' && <><button className={linkClass} onClick={() => changeMode('forgot')}>Forgot password?</button><p>New to DoubtDesk? <button className={`font-semibold text-[#111114] ${linkClass}`} onClick={() => changeMode('signup')}>Create an account</button></p><button className={linkClass} onClick={() => changeMode('check-email')}>Resend verification email</button></>}
                  {mode !== 'login' && <button className={`font-semibold text-[#111114] ${linkClass}`} onClick={() => changeMode('login')}>Back to sign in</button>}
                  {mode === 'forgot' && <button className={linkClass} onClick={() => changeMode('check-email')}>Resend verification email instead</button>}
                  {mode === 'check-email' && <button className={linkClass} onClick={() => changeMode('forgot')}>Forgot your password?</button>}
                </div>
                <p className="mt-4 flex items-start gap-2 text-xs leading-5 text-[#6E6E78]"><LockKeyhole size={13} className="mt-[3px] shrink-0" />Secure session · Your PDFs belong to you. Only you can see them.</p>
              </section>
            </div>
          </div>
        </section>

        <section className="relative overflow-hidden pb-14 pt-[110px]">
          <SunDoodle className="float left-[6%] top-14 w-[90px] sm:w-[120px]" />
          <Sparks className="right-[12%] top-24 w-[50px]" />
          <Cross3D depth={24} className="float float-2 -right-8 top-[300px] w-[160px]" style={{ '--r': '12deg' } as CSSProperties} />
          <div className="relative z-[2] mx-auto max-w-[1200px] px-4 sm:px-6">
            <h2 ref={beliefRef} className={`mx-auto max-w-[980px] text-balance text-center text-[clamp(30px,4.6vw,58px)] font-medium leading-[1.2] tracking-[-0.03em] ${beliefShown ? 'is-revealed' : ''}`}>
              {beliefWords.map((word, index) => (
                <span key={`${word}-${index}`}><span className={index < 2 ? 'reveal-keep' : 'reveal-word'} style={{ animationDelay: `${index * 70}ms` }}>{word}</span>{index < beliefWords.length - 1 ? ' ' : ''}</span>
              ))}
            </h2>
            <div className="mt-16 flex flex-wrap justify-center gap-6">
              {STATS.map(stat => (
                <div key={stat.value} className="scroll-in flex h-[148px] w-[148px] flex-col items-center justify-center rounded-full bg-[radial-gradient(circle_at_50%_38%,#fff,#EBEBE9_72%)] p-5 text-center transition-transform duration-[400ms] ease-[cubic-bezier(.3,1.4,.5,1)] hover:-rotate-3 hover:scale-[1.07] sm:h-[180px] sm:w-[180px]">
                  <b className="text-[30px] font-semibold tracking-[-0.03em] sm:text-[38px]">{stat.value}</b>
                  <span className="mt-0.5 text-[12.5px] leading-[1.45] text-muted">{stat.text}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="how" aria-label="Example answer" className="scroll-mt-6 py-20">
          <div className="mx-auto max-w-[1200px] px-4 sm:px-6">
            <SectionTitle muted="What an answer" strong="looks like" />
            <div className="mt-12 flex flex-wrap items-center gap-10">
              <div className="scroll-in relative flex min-w-0 flex-[999_1_540px] flex-wrap gap-[18px] rounded-[32px] bg-white p-[22px] shadow-[0_30px_70px_rgba(17,17,20,.06)]" aria-hidden="true">
                <Capsule3D depth={14} className="float float-3 -bottom-8 -right-8 w-[110px]" style={{ '--r': '-20deg' } as CSSProperties} />
                <div className="min-w-0 flex-[999_1_280px] p-2.5">
                  <p className="text-[13px] text-muted">Q · How does the nephron filter blood?</p>
                  <p className="stamp stamp--grounded mt-3"><Check size={11} strokeWidth={3} />From documents</p>
                  <p className="mt-3.5 text-base leading-[1.75] text-body">Blood enters the glomerulus under high pressure, which forces water and small solutes into Bowman’s capsule <span className="cite-marker">1</span>. The filtrate then passes along the tubule, where glucose and most water are reabsorbed <span className="cite-marker">2</span>.</p>
                </div>
                <div className="flex min-w-0 flex-[1_1_200px] flex-col gap-2 rounded-[22px] bg-[#F6F6F4] p-[18px]">
                  <p className="mb-1 text-xs text-muted">biology-ch7.pdf · p. 112</p>
                  <span className="block h-[9px] w-[70%] rounded-full bg-ink" />
                  <span className="page-line w-full" /><span className="page-line w-[92%]" />
                  <span className="glow flex flex-col gap-1.5 rounded-[10px] bg-mint-soft p-[7px]"><span className="page-line w-full !bg-[#7FD6A6]" /><span className="page-line w-[86%] !bg-[#7FD6A6]" /><span className="page-line w-[92%] !bg-[#7FD6A6]" /></span>
                  <span className="page-line w-[96%]" /><span className="page-line w-[64%]" />
                  <span className="mt-1.5 block h-16 w-full rounded-[10px] border border-dashed border-[#D5D5D2]" />
                </div>
              </div>
              <div className="flex min-w-0 flex-[1_1_280px] flex-col gap-3">
                {FEATURES.map((item, index) => (
                  <button key={item.title} type="button" onClick={() => setFeature(index)} aria-expanded={feature === index}
                    className={`rounded-3xl px-[18px] py-3.5 text-left transition-colors duration-300 ${feature === index ? 'bg-white' : 'hover:bg-white/60'}`}>
                    <span className={`block text-[21px] font-semibold tracking-[-0.015em] transition-colors duration-500 ${feature === index ? 'text-ink' : 'text-ghost'}`}>{item.title}</span>
                    <span className={`block overflow-hidden text-[15px] leading-relaxed text-muted transition-all duration-500 ${feature === index ? 'mt-1.5 max-h-32 opacity-100' : 'max-h-0 opacity-0'}`}>{item.text}</span>
                    {feature === index && <span className="mt-3 block h-[3px] overflow-hidden rounded bg-rule"><span key={feature} className="feature-progress block h-full bg-purple" /></span>}
                  </button>
                ))}
              </div>
            </div>
          </div>
        </section>

        <section aria-label="How it works" className="pb-24 pt-5">
          <div className="mx-auto max-w-[1200px] px-4 sm:px-6">
            <SectionTitle strong={<>Three steps, <span className="text-ghost">no setup.</span></>} />
            <ol className="mt-11 grid grid-cols-[repeat(auto-fit,minmax(260px,1fr))] gap-[18px]">
              {STEPS.map((step, index) => (
                <li key={step.title} className="lift scroll-in group relative min-h-[230px] overflow-hidden rounded-[30px] bg-white p-[30px] hover:shadow-[0_24px_50px_rgba(17,17,20,.08)]">
                  {index === 1 && <StarDoodle className="right-4 top-3.5 w-[84px]" />}
                  <span className="grid h-[46px] w-[46px] place-items-center rounded-full bg-night font-semibold text-white transition-colors group-hover:bg-purple">{index + 1}</span>
                  <h3 className="mb-2 mt-[22px] text-[22px] font-semibold tracking-[-0.02em]">{step.title}</h3>
                  <p className="text-[15px] leading-relaxed text-muted">{step.text}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        <section id="labels" className="relative scroll-mt-6 overflow-hidden rounded-[30px] bg-night py-[90px] text-white sm:rounded-[44px]">
          <Blob3D depth={16} className="float -left-10 top-10 w-[170px]" />
          <Star3D depth={26} className="float float-2 -right-5 bottom-8 w-[150px]" style={{ '--r': '-14deg' } as CSSProperties} />
          <div className="relative z-[2] mx-auto max-w-[1200px] px-4 sm:px-6">
            <div className="mb-6 text-center"><span className="pill pill--lime"><span className="pill-dot" />Answer labels</span></div>
            <SectionTitle dark center strong="It says where an answer came from." />
            <p className="mx-auto mt-5 max-w-[520px] text-center text-[17px] leading-relaxed text-night-body">Each reply carries one of four labels, so you always know how far to trust it.</p>
            <div className="mt-14 grid grid-cols-[repeat(auto-fit,minmax(240px,1fr))] gap-4">
              {LABELS.map(item => (
                <div key={item.label} className="scroll-in group rounded-[26px] border border-night-rule bg-night-2 p-[26px] transition hover:-translate-y-1 hover:border-[#44444C]">
                  <span className={`stamp ${item.stamp} transition-transform group-hover:-rotate-3 group-hover:scale-105`}>{item.label}</span>
                  <p className="mt-4 text-[14.5px] leading-relaxed text-[#A9A9B2]">{item.text}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="plans" className="scroll-mt-6">
          <div className="mx-auto max-w-[1200px] px-4 py-[110px] sm:px-6">
            <SectionTitle strong={<>Start free. <span className="text-ghost">Upgrade for more.</span></>} />
            <p className="mt-4 text-sm text-muted">Limits reset on a rolling 24-hour window.</p>
            <div className="mt-12 flex flex-wrap items-stretch gap-[18px]">
              {PLANS.map(plan => (
                <article key={plan.name} className={`lift scroll-in relative flex min-w-0 flex-[1_1_280px] flex-col overflow-hidden rounded-[32px] p-8 ${plan.featured ? 'bg-night text-white' : 'bg-white'}`}>
                  {plan.featured && <Star3D className="float -right-7 -top-7 w-[120px]" />}
                  <p className={`flex items-center justify-between text-sm font-medium ${plan.featured ? 'text-[#A9A9B2]' : 'text-muted'}`}>
                    <span>{plan.name}</span>
                    {plan.featured && <span className="pill pill--lime mr-16 !px-2.5 !py-1 !text-[11.5px]">Recommended</span>}
                  </p>
                  <p className="mt-3.5 text-[46px] font-semibold tracking-[-0.04em]">{plan.price}{plan.period && <span className={`text-[15px] font-normal tracking-normal ${plan.featured ? 'text-night-body' : 'text-muted'}`}> {plan.period}</span>}</p>
                  <dl className="mt-5 flex-1 text-[14.5px]">
                    {[['Questions a day', plan.questions], ['Tokens a day', plan.tokens], ['Questions at once', plan.concurrent]].map(([term, value], index) => (
                      <div key={term} className={`flex justify-between gap-3 py-3 ${index < 2 ? `border-b ${plan.featured ? 'border-[#26262C]' : 'border-rule'}` : ''}`}>
                        <dt>{term}</dt><dd className="font-semibold tabular-nums">{value}</dd>
                      </div>
                    ))}
                  </dl>
                  {plan.name === 'Free'
                    ? <a href="#account" onClick={() => openAccount('signup')} className="secondary-button mt-6 h-[50px]">Create an account</a>
                    : <button type="button" disabled className={`mt-6 h-[50px] rounded-full border border-dashed font-semibold ${plan.featured ? 'border-[#44444C] text-night-muted' : 'border-ghost text-muted'}`}>Coming soon</button>}
                </article>
              ))}
            </div>
            <p className="mt-6 text-[14.5px] text-muted">Every plan: PDFs up to 25 MB, 30 saved chats, page citations and answer labels.</p>
          </div>
        </section>
      </main>

      <footer className="relative overflow-hidden rounded-t-[30px] bg-night pb-32 pt-[90px] text-white sm:rounded-t-[44px]">
        <Star3D depth={30} className="float right-[4%] top-10 w-[180px] sm:w-[300px]" style={{ '--r': '-8deg' } as CSSProperties} />
        <Blob3D depth={18} className="float float-2 -right-5 top-[220px] hidden w-[160px] sm:block" />
        <Sphere3D depth={10} className="float float-3 right-[30%] top-[250px] w-[90px]" />
        <div className="relative z-[2] mx-auto max-w-[1200px] px-4 sm:px-6">
          <h2 className="scroll-in max-w-[640px] text-balance text-[clamp(38px,5.6vw,74px)] font-semibold leading-[1.05] tracking-[-0.04em]">Clear your STEM doubts from your own books.</h2>
          <a href="#account" onClick={() => openAccount('signup')} className="group mt-8 inline-flex items-center gap-3 border-b-2 border-white pb-1 text-xl font-medium">
            Get started <span className="grid h-[30px] w-[30px] place-items-center rounded-full border-2 border-white transition group-hover:-rotate-45 group-hover:bg-white group-hover:text-black"><ArrowRight size={15} /></span>
          </a>
          <div className="mt-20 flex flex-wrap justify-between gap-10 border-t border-[#222228] pt-8">
            <div className="max-w-[420px] flex-[1_1_300px]">
              <Wordmark size={34} tone="dark" textClassName="text-xl" />
              <p className="mt-4 leading-relaxed text-night-body">Answers from your own study material, with the page to prove it.</p>
            </div>
            <div className="flex flex-wrap gap-16 text-[15px]">
              <nav aria-label="Product" className="flex flex-col gap-3">
                <span className="night-label">Product</span>
                {NAV.map(item => <a key={item.id} href={item.href} className="text-night-body transition hover:text-white">{item.label}</a>)}
              </nav>
              <nav aria-label="Account links" className="flex flex-col gap-3">
                <span className="night-label">Account</span>
                <a href="#account" onClick={() => openAccount('login')} className="text-night-body transition hover:text-white">Sign in</a>
                <a href="#account" onClick={() => openAccount('signup')} className="text-night-body transition hover:text-white">Create an account</a>
                <a href="#account" onClick={() => openAccount('forgot')} className="text-night-body transition hover:text-white">Forgot password?</a>
              </nav>
            </div>
          </div>
          <p className="mt-12 text-xs text-night-muted">© {COPYRIGHT_YEAR} DoubtDesk</p>
        </div>
      </footer>

      <nav aria-label="Main" className="pill-nav flex max-w-[calc(100vw-32px)] gap-0.5 rounded-full border border-[#26262C] bg-night/90 p-1.5 shadow-[0_18px_40px_rgba(0,0,0,.35)] backdrop-blur-md">
        {NAV.map(item => (
          <a key={item.id} href={item.href} className={`whitespace-nowrap rounded-full px-4 py-2 text-[13px] transition ${item.id === 'labels' ? 'hidden sm:inline-block' : ''} ${activeSection === item.id ? 'bg-[#26262C] text-white' : 'text-night-body hover:text-white'}`}>{item.label}</a>
        ))}
        <a href="#account" onClick={() => openAccount('login')} className="whitespace-nowrap rounded-full bg-white px-4 py-2 text-[13px] font-semibold text-black">Sign in</a>
      </nav>
    </div>
  )
}

import { useEffect, useRef, useState } from 'react'
import type { FormEvent } from 'react'
import { ArrowRight, BookOpen, GraduationCap, LoaderCircle, LockKeyhole, Mail } from 'lucide-react'
import { authApi } from './api'
import type { AuthUser } from './api'

type Mode = 'login' | 'signup' | 'forgot' | 'verify' | 'reset' | 'check-email'

function initialRoute(): { mode: Mode; token: string } {
  const path = window.location.pathname.toLowerCase()
  const params = new URLSearchParams(window.location.search)
  const token = params.get('token') || ''
  if (path.endsWith('/verify-email')) return { mode: 'verify', token }
  if (path.endsWith('/reset-password')) return { mode: 'reset', token }
  return { mode: 'login', token: '' }
}

export function AuthScreen({ onLogin }: { onLogin: (user: AuthUser) => void }) {
  const initial = useRef(initialRoute())
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

  useEffect(() => {
    if (initial.current.token) {
      window.history.replaceState({}, '', '/')
    }
  }, [])

  function changeMode(next: Mode) {
    setMode(next)
    setError('')
    setNotice('')
    setPassword('')
    setConfirmPassword('')
    if (next === 'login' && identifier === '' && email) setIdentifier(email)
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

  return (
    <div className="min-h-dvh bg-[#F8FAFC] px-4 py-9 text-[#0F172A] sm:py-14">
      <div className="mx-auto max-w-md">
        <div className="mb-8 flex items-center justify-center gap-3">
          <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-[#4169FF] text-white shadow-sm"><GraduationCap size={25}/></div>
          <span className="text-2xl font-bold tracking-tight">DoubtDesk</span>
        </div>
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm sm:p-9">
          <div className="mb-6 flex h-11 w-11 items-center justify-center rounded-xl bg-blue-50 text-[#4169FF]"><BookOpen size={22} /></div>
          <h1 className="text-2xl font-bold tracking-tight">{titles[mode]}</h1>
          <p className="mt-2 text-sm leading-6 text-slate-500">{descriptions[mode]}</p>

          <form onSubmit={(event) => void submit(event)} className="mt-7 space-y-4">
            {mode === 'signup' && <label className="block text-sm font-medium text-slate-700">Username
              <input className="field mt-1.5 w-full" aria-label="Username" autoComplete="username" required minLength={3} maxLength={32} pattern="[A-Za-z0-9_]+" value={username} onChange={(event) => setUsername(event.target.value)} placeholder="your_username" />
            </label>}
            {(mode === 'signup' || mode === 'forgot' || mode === 'check-email') && <label className="block text-sm font-medium text-slate-700">Email
              <input className="field mt-1.5 w-full" type="email" aria-label="Email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} placeholder="you@example.com" />
            </label>}
            {mode === 'login' && <label className="block text-sm font-medium text-slate-700">Username or email
              <input className="field mt-1.5 w-full" aria-label="Username or email" autoComplete="username" required value={identifier} onChange={(event) => setIdentifier(event.target.value)} placeholder="Username or email" />
            </label>}
            {(mode === 'signup' || mode === 'login' || mode === 'reset') && <label className="block text-sm font-medium text-slate-700">{mode === 'reset' ? 'New password' : 'Password'}
              <input className="field mt-1.5 w-full" type="password" aria-label={mode === 'reset' ? 'New password' : 'Password'} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={mode === 'login' ? 1 : 8} maxLength={128} value={password} onChange={(event) => setPassword(event.target.value)} placeholder={mode === 'login' ? 'Enter your password' : 'At least 8 characters'} />
            </label>}
            {(mode === 'signup' || mode === 'reset') && <label className="block text-sm font-medium text-slate-700">Confirm password
              <input className="field mt-1.5 w-full" type="password" aria-label="Confirm password" autoComplete="new-password" required minLength={8} maxLength={128} value={confirmPassword} onChange={(event) => setConfirmPassword(event.target.value)} placeholder="Repeat your password" />
            </label>}
            {mode === 'verify' && <div className="flex items-center gap-2 rounded-xl bg-blue-50 p-3 text-sm text-blue-800"><Mail size={18} />{token ? 'Verification link received.' : 'No verification token was found in this link.'}</div>}
            {mode === 'reset' && !token && <p role="alert" className="text-sm text-red-700">No reset token was found. Request a new link.</p>}
            {notice && <p role="status" className="rounded-xl bg-blue-50 px-3 py-2 text-sm leading-6 text-blue-800">{notice}</p>}
            {error && <p role="alert" className="rounded-xl bg-red-50 px-3 py-2 text-sm leading-6 text-red-700">{error}</p>}
            <button className="primary-button w-full justify-center" type="submit" disabled={busy || ((mode === 'verify' || mode === 'reset') && !token)}>
              {busy && <LoaderCircle className="animate-spin" size={17} />}
              {{ login: 'Sign in', signup: 'Create account', forgot: 'Send reset link', verify: 'Verify email', reset: 'Reset password', 'check-email': 'Resend verification email' }[mode]}
              {!busy && <ArrowRight size={17} />}
            </button>
          </form>
          <div className="mt-6 flex flex-col items-center gap-3 text-sm">
            {mode === 'login' && <><button className="text-[#3155E7] hover:underline" onClick={() => changeMode('forgot')}>Forgot password?</button><p className="text-slate-500">New to DoubtDesk? <button className="font-semibold text-[#3155E7] hover:underline" onClick={() => changeMode('signup')}>Create an account</button></p><button className="text-slate-500 hover:underline" onClick={() => changeMode('check-email')}>Resend verification email</button></>}
            {mode !== 'login' && <button className="font-semibold text-[#3155E7] hover:underline" onClick={() => changeMode('login')}>Back to sign in</button>}
            {mode === 'forgot' && <button className="text-slate-500 hover:underline" onClick={() => changeMode('check-email')}>Resend verification email instead</button>}
            {mode === 'check-email' && <button className="text-slate-500 hover:underline" onClick={() => changeMode('forgot')}>Forgot your password?</button>}
          </div>
        </section>
        <p className="mt-6 text-center text-xs leading-5 text-slate-400"><LockKeyhole size={13} className="mr-1 inline" />Secure session · Your PDFs are available only to authorized accounts</p>
      </div>
    </div>
  )
}

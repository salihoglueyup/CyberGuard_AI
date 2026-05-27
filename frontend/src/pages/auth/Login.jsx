import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Shield, Lock, Eye, EyeOff, Loader2, User } from 'lucide-react';
import { useToast } from '../../components/ui/Toast';
import { DataMatrix, TypewriterText } from '../../components/hud';
import { API_BASE } from '../../services/api';

// Boot sequence lines
const BOOT_LINES = [
    '[INIT] CyberGuard AI v3.0.0',
    '[CORE] Neural defense engine ... OK',
    '[NET]  Threat intelligence feed ... CONNECTED',
    '[AI]   SSA-LSTM IDS model loaded ... OK',
    '[SYS]  Honeypot network ... 12 nodes active',
    '[SEC]  Encryption layer ... AES-256-GCM',
    '[BOOT] System ready. Awaiting authentication.',
];

export default function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const [bootDone, setBootDone] = useState(false);
    const [bootLines, setBootLines] = useState([]);
    const navigate = useNavigate();
    const toast = useToast();

    // Boot sequence animation
    useEffect(() => {
        let idx = 0;
        const id = setInterval(() => {
            if (idx < BOOT_LINES.length) {
                setBootLines(prev => [...prev, BOOT_LINES[idx]]);
                idx++;
            } else {
                clearInterval(id);
                setTimeout(() => setBootDone(true), 400);
            }
        }, 250);
        return () => clearInterval(id);
    }, []);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!username || !password) {
            toast.warning('Tum alanlari doldurun');
            return;
        }
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/auth/login`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password }),
            });
            const data = await res.json();
            if (data.success) {
                sessionStorage.setItem('token', data.data.token);
                sessionStorage.setItem('user', JSON.stringify(data.data.user));
                toast.success('Giris basarili!');
                navigate('/');
            } else {
                toast.error(data.detail || data.error || 'Giris basarisiz');
            }
        } catch (error) {
            toast.error('Baglanti hatasi: ' + error.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] flex items-center justify-center p-4 overflow-hidden">
            {/* Matrix background */}
            <div className="absolute inset-0 opacity-30">
                <DataMatrix />
            </div>

            {/* Grid overlay */}
            <div className="absolute inset-0" style={{
                backgroundImage: 'linear-gradient(rgba(56,189,248,0.02) 1px, transparent 1px), linear-gradient(90deg, rgba(56,189,248,0.02) 1px, transparent 1px)',
                backgroundSize: '60px 60px',
            }} />

            {/* Vignette */}
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_30%,rgba(0,0,0,0.7))]" />

            <div className="relative z-10 w-full max-w-sm">
                {/* Boot sequence */}
                {!bootDone && (
                    <div className="mb-6 font-mono text-[10px] text-[var(--hud-emerald)] space-y-0.5 max-h-40 overflow-hidden">
                        {bootLines.map((line, i) => (
                            <div key={i} className="fade-in">{line}</div>
                        ))}
                        {bootLines.length < BOOT_LINES.length && (
                            <span className="inline-block w-2 h-3 bg-[var(--hud-emerald)] animate-pulse" />
                        )}
                    </div>
                )}

                {/* Logo */}
                <div className={`text-center mb-6 transition-all duration-500 ${bootDone ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                    <div className="w-14 h-14 mx-auto rounded-md border border-[rgba(56,189,248,0.3)] flex items-center justify-center bg-[rgba(56,189,248,0.04)] mb-3">
                        <Shield className="w-8 h-8 text-[var(--hud-cyan)]" />
                    </div>
                    <h1 className="text-lg font-bold font-mono text-[var(--hud-cyan)] tracking-wide">CyberGuard</h1>
                    <p className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mt-1">AI Command Center</p>
                </div>

                {/* Login form */}
                <div className={`transition-all duration-500 delay-200 ${bootDone ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}>
                    <div className="relative p-6 bg-[var(--hud-surface)] border border-[var(--hud-border)] rounded">
                        <form onSubmit={handleSubmit} className="space-y-4">
                            {/* Username */}
                            <div>
                                <label className="block text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mb-1.5">
                                    Kullanıcı Adı
                                </label>
                                <div className="relative">
                                    <User className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--hud-text-dim)]" />
                                    <input
                                        type="text"
                                        value={username}
                                        onChange={(e) => setUsername(e.target.value)}
                                        placeholder="admin"
                                        className="w-full pl-8 pr-3 py-2 bg-[rgba(56,189,248,0.02)] border border-[var(--hud-border)] rounded text-sm text-[var(--hud-text)] placeholder:text-[var(--hud-text-dim)] focus:outline-none focus:border-[var(--hud-cyan)] focus:shadow-[0_0_12px_rgba(56,189,248,0.1)] transition-all font-mono"
                                    />
                                </div>
                            </div>

                            {/* Password */}
                            <div>
                                <label className="block text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wide mb-1.5">
                                    Şifre
                                </label>
                                <div className="relative">
                                    <Lock className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-[var(--hud-text-dim)]" />
                                    <input
                                        type={showPassword ? 'text' : 'password'}
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        placeholder="********"
                                        className="w-full pl-8 pr-10 py-2 bg-[rgba(56,189,248,0.02)] border border-[var(--hud-border)] rounded text-sm text-[var(--hud-text)] placeholder:text-[var(--hud-text-dim)] focus:outline-none focus:border-[var(--hud-cyan)] focus:shadow-[0_0_12px_rgba(56,189,248,0.1)] transition-all font-mono"
                                    />
                                    <button
                                        type="button"
                                        onClick={() => setShowPassword(!showPassword)}
                                        className="absolute right-2.5 top-1/2 -translate-y-1/2 text-[var(--hud-text-dim)] hover:text-[var(--hud-text)]"
                                    >
                                        {showPassword ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                                    </button>
                                </div>
                            </div>

                            {/* Submit */}
                            <button
                                type="submit"
                                disabled={loading}
                                className="w-full py-2.5 bg-[rgba(56,189,248,0.08)] border border-[rgba(56,189,248,0.3)] hover:bg-[rgba(56,189,248,0.15)] hover:border-[rgba(56,189,248,0.5)] hover:shadow-[0_0_20px_rgba(56,189,248,0.15)] text-[var(--hud-cyan)] font-mono text-xs tracking-wide rounded transition-all disabled:opacity-40 flex items-center justify-center gap-2"
                            >
                                {loading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Doğrulanıyor...
                                    </>
                                ) : (
                                    'Giriş Yap'
                                )}
                            </button>
                        </form>

                        {/* Register link */}
                        <div className="mt-4 text-center text-[10px] font-mono text-[var(--hud-text-dim)]">
                            Hesabiniz yok mu?{' '}
                            <Link to="/register" className="text-[var(--hud-cyan)] hover:underline">
                                Kayıt Ol
                            </Link>
                        </div>
                    </div>

                    {/* Demo credentials */}
                    {import.meta.env.DEV && (
                        <div className="mt-4 p-2.5 bg-[rgba(56,189,248,0.02)] border border-[var(--hud-border)] rounded text-center">
                            <p className="text-[9px] font-mono text-[var(--hud-text-dim)] tracking-wider">Demo Hesabı</p>
                            <p className="text-[11px] font-mono text-[var(--hud-cyan)] mt-0.5">admin / admin123</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

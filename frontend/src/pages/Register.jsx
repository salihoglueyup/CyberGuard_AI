import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Shield, Mail, Lock, User, Eye, EyeOff, Loader2, Check } from 'lucide-react';
import { useToast } from '../components/ui/Toast';

const API_BASE = 'http://localhost:8000/api';

export default function Register() {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        password_confirm: ''
    });
    const [showPassword, setShowPassword] = useState(false);
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const toast = useToast();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        // Validasyon
        if (!formData.username || !formData.email || !formData.password || !formData.password_confirm) {
            toast.warning('Lütfen tüm alanları doldurun');
            return;
        }

        if (formData.password !== formData.password_confirm) {
            toast.error('Şifreler eşleşmiyor');
            return;
        }

        if (formData.password.length < 6) {
            toast.error('Şifre en az 6 karakter olmalı');
            return;
        }

        setLoading(true);

        try {
            const res = await fetch(`${API_BASE}/auth/register`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData),
            });

            const data = await res.json();

            if (data.success) {
                // Token'ı kaydet
                localStorage.setItem('token', data.data.token);
                localStorage.setItem('user', JSON.stringify(data.data.user));

                toast.success('Kayıt başarılı!');
                navigate('/');
            } else {
                toast.error(data.detail || data.error || 'Kayıt başarısız');
            }
        } catch (error) {
            toast.error('Bağlantı hatası: ' + error.message);
        } finally {
            setLoading(false);
        }
    };

    // Şifre gücü kontrolü
    const getPasswordStrength = () => {
        const pass = formData.password;
        if (!pass) return { score: 0, label: '', color: '' };

        let score = 0;
        if (pass.length >= 6) score++;
        if (pass.length >= 8) score++;
        if (/[a-z]/.test(pass) && /[A-Z]/.test(pass)) score++;
        if (/\d/.test(pass)) score++;
        if (/[^a-zA-Z0-9]/.test(pass)) score++;

        if (score <= 2) return { score, label: 'Zayıf', color: 'bg-red-500' };
        if (score <= 3) return { score, label: 'Orta', color: 'bg-yellow-500' };
        return { score, label: 'Güçlü', color: 'bg-green-500' };
    };

    const strength = getPasswordStrength();

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
            <div className="w-full max-w-md">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center mb-4">
                        <Shield className="w-10 h-10 text-white" />
                    </div>
                    <h1 className="text-3xl font-bold text-white">CyberGuard AI</h1>
                    <p className="text-slate-400 mt-2">Yeni hesap oluşturun</p>
                </div>

                {/* Form Card */}
                <div className="bg-slate-800/50 backdrop-blur-xl rounded-2xl border border-slate-700/50 p-8">
                    <form onSubmit={handleSubmit} className="space-y-5">
                        {/* Username */}
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                Kullanıcı Adı
                            </label>
                            <div className="relative">
                                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type="text"
                                    name="username"
                                    value={formData.username}
                                    onChange={handleChange}
                                    placeholder="kullaniciadi"
                                    className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                            </div>
                        </div>

                        {/* Email */}
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                E-posta
                            </label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type="email"
                                    name="email"
                                    value={formData.email}
                                    onChange={handleChange}
                                    placeholder="ornek@email.com"
                                    className="w-full pl-10 pr-4 py-3 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                            </div>
                        </div>

                        {/* Password */}
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                Şifre
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    name="password"
                                    value={formData.password}
                                    onChange={handleChange}
                                    placeholder="••••••••"
                                    className="w-full pl-10 pr-12 py-3 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                                <button
                                    type="button"
                                    onClick={() => setShowPassword(!showPassword)}
                                    className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                                >
                                    {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                                </button>
                            </div>
                            {/* Password Strength */}
                            {formData.password && (
                                <div className="mt-2">
                                    <div className="flex gap-1">
                                        {[1, 2, 3, 4, 5].map((i) => (
                                            <div
                                                key={i}
                                                className={`h-1 flex-1 rounded-full ${i <= strength.score ? strength.color : 'bg-slate-700'}`}
                                            />
                                        ))}
                                    </div>
                                    <p className={`text-xs mt-1 ${strength.score <= 2 ? 'text-red-400' : strength.score <= 3 ? 'text-yellow-400' : 'text-green-400'}`}>
                                        {strength.label}
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Confirm Password */}
                        <div>
                            <label className="block text-sm font-medium text-slate-300 mb-2">
                                Şifre Tekrar
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-slate-400" />
                                <input
                                    type={showPassword ? 'text' : 'password'}
                                    name="password_confirm"
                                    value={formData.password_confirm}
                                    onChange={handleChange}
                                    placeholder="••••••••"
                                    className="w-full pl-10 pr-12 py-3 bg-slate-900/50 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-blue-500 transition-colors"
                                />
                                {formData.password_confirm && formData.password === formData.password_confirm && (
                                    <Check className="absolute right-3 top-1/2 -translate-y-1/2 w-5 h-5 text-green-500" />
                                )}
                            </div>
                        </div>

                        {/* Terms */}
                        <label className="flex items-start gap-2 text-sm text-slate-400">
                            <input type="checkbox" className="rounded bg-slate-700 border-slate-600 mt-0.5" required />
                            <span>
                                <a href="#" className="text-blue-400 hover:text-blue-300">Kullanım şartlarını</a> ve{' '}
                                <a href="#" className="text-blue-400 hover:text-blue-300">gizlilik politikasını</a> kabul ediyorum.
                            </span>
                        </label>

                        {/* Submit */}
                        <button
                            type="submit"
                            disabled={loading}
                            className="w-full py-3 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-white font-medium rounded-xl transition-all disabled:opacity-50 flex items-center justify-center gap-2"
                        >
                            {loading ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Kayıt yapılıyor...
                                </>
                            ) : (
                                'Kayıt Ol'
                            )}
                        </button>
                    </form>

                    {/* Login Link */}
                    <div className="mt-6 text-center text-sm text-slate-400">
                        Zaten hesabınız var mı?{' '}
                        <Link to="/login" className="text-blue-400 hover:text-blue-300 font-medium">
                            Giriş yapın
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    );
}

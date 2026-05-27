import { useTranslation } from 'react-i18next';
import { Globe } from 'lucide-react';

const LANGUAGES = [
    { code: 'tr', label: 'Türkçe', flag: '🇹🇷' },
    { code: 'en', label: 'English', flag: '🇬🇧' },
];

export default function LanguageSwitcher({ compact = false }) {
    const { i18n } = useTranslation();
    const current = i18n.language?.startsWith('tr') ? 'tr' : 'en';

    const toggle = () => {
        const next = current === 'tr' ? 'en' : 'tr';
        i18n.changeLanguage(next);
    };

    if (compact) {
        return (
            <button
                onClick={toggle}
                className="flex items-center gap-1.5 px-2 py-1 rounded border border-[var(--hud-border)] text-[10px] font-mono text-[var(--hud-text-muted)] hover:text-[var(--hud-cyan)] hover:border-[rgba(56,189,248,0.3)] transition-all"
                title={`Switch to ${current === 'tr' ? 'English' : 'Türkçe'}`}
            >
                <Globe className="w-3 h-3" />
                {current.toUpperCase()}
            </button>
        );
    }

    return (
        <div className="flex items-center gap-2">
            <Globe className="w-4 h-4 text-[var(--hud-text-dim)]" />
            <div className="flex rounded border border-[var(--hud-border)] overflow-hidden">
                {LANGUAGES.map((lang) => (
                    <button
                        key={lang.code}
                        onClick={() => i18n.changeLanguage(lang.code)}
                        className={`flex items-center gap-1.5 px-3 py-1.5 text-[10px] font-mono transition-all ${
                            current === lang.code
                                ? 'bg-[rgba(56,189,248,0.1)] text-[var(--hud-cyan)] border-r border-[var(--hud-border)]'
                                : 'text-[var(--hud-text-muted)] hover:bg-[rgba(56,189,248,0.04)]'
                        }`}
                    >
                        <span>{lang.flag}</span>
                        {lang.label}
                    </button>
                ))}
            </div>
        </div>
    );
}

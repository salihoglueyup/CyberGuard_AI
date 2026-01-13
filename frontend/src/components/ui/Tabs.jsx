import { useState } from 'react';

export default function Tabs({
    tabs,
    defaultTab,
    onChange,
    variant = 'default',
    className = '',
}) {
    const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

    const handleTabChange = (tabId) => {
        setActiveTab(tabId);
        onChange?.(tabId);
    };

    const variants = {
        default: 'tabs',
        pills: 'flex gap-2',
        underline: 'flex gap-4 border-b border-slate-700',
    };

    const tabStyles = {
        default: (isActive) => `tab ${isActive ? 'active' : ''}`,
        pills: (isActive) => `
      px-4 py-2 rounded-lg text-sm font-medium transition-all
      ${isActive
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:text-white hover:bg-slate-800'
            }
    `,
        underline: (isActive) => `
      pb-3 text-sm font-medium transition-colors border-b-2 -mb-px
      ${isActive
                ? 'text-blue-400 border-blue-400'
                : 'text-slate-400 hover:text-white border-transparent'
            }
    `,
    };

    const activeTabContent = tabs.find(t => t.id === activeTab);

    return (
        <div className={className}>
            <div className={variants[variant]}>
                {tabs.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => handleTabChange(tab.id)}
                        className={tabStyles[variant](activeTab === tab.id)}
                    >
                        <div className="flex items-center gap-2">
                            {tab.icon && <tab.icon className="w-4 h-4" />}
                            {tab.label}
                            {tab.badge && (
                                <span className="ml-1 px-1.5 py-0.5 text-[10px] bg-blue-500/20 text-blue-400 rounded-full">
                                    {tab.badge}
                                </span>
                            )}
                        </div>
                    </button>
                ))}
            </div>
            {activeTabContent?.content && (
                <div className="mt-4">
                    {activeTabContent.content}
                </div>
            )}
        </div>
    );
}

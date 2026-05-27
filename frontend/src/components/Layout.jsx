import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import { ToastContainer, CriticalAlertModal } from './ui/Toast';
import { useSidebarStore } from '../store';
import { CommandBar } from './hud';

export default function Layout() {
    const collapsed = useSidebarStore((s) => s.collapsed);
    return (
        <div className="relative min-h-screen bg-[var(--hud-bg)] text-[var(--hud-text)] overflow-hidden">
            {/* Ctrl+K command palette */}
            <CommandBar />

            <Sidebar />

            <main className={`relative z-10 flex flex-col min-h-screen transition-all duration-300 ${collapsed ? 'ml-16' : 'ml-56'}`}>
                <Header />
                <div className="flex-1 p-6 overflow-y-auto">
                    <Outlet />
                </div>
            </main>

            <ToastContainer />
            <CriticalAlertModal />
        </div>
    );
}

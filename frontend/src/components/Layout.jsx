import { Outlet } from 'react-router-dom';
import Sidebar from './Sidebar';
import Header from './Header';
import { ToastContainer } from './ui/Toast';

export default function Layout() {
    return (
        <div className="flex min-h-screen bg-slate-900">
            <Sidebar />
            <main className="flex-1 ml-64 transition-all duration-300">
                <Header />
                <div className="p-6">
                    <Outlet />
                </div>
            </main>
            <ToastContainer />
        </div>
    );
}

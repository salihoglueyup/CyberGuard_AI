import { motion, AnimatePresence } from 'framer-motion';

// --- Fade-in wrapper with stagger support ---
export function FadeIn({ children, delay = 0, duration = 0.4, className = '', ...props }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration, delay, ease: 'easeOut' }}
            className={className}
            {...props}
        >
            {children}
        </motion.div>
    );
}

// --- Staggered children container ---
export function StaggerContainer({ children, stagger = 0.06, className = '' }) {
    return (
        <motion.div
            initial="hidden"
            animate="visible"
            variants={{
                hidden: {},
                visible: { transition: { staggerChildren: stagger } },
            }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

export function StaggerItem({ children, className = '' }) {
    return (
        <motion.div
            variants={{
                hidden: { opacity: 0, y: 16 },
                visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: 'easeOut' } },
            }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// --- Scale-in (for cards, modals) ---
export function ScaleIn({ children, delay = 0, className = '' }) {
    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.92 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3, delay, ease: [0.23, 1, 0.32, 1] }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// --- Slide from direction ---
export function SlideIn({ children, from = 'left', delay = 0, className = '' }) {
    const axis = from === 'left' || from === 'right' ? 'x' : 'y';
    const value = from === 'left' || from === 'top' ? -30 : 30;
    return (
        <motion.div
            initial={{ opacity: 0, [axis]: value }}
            animate={{ opacity: 1, [axis]: 0 }}
            exit={{ opacity: 0, [axis]: value / 2 }}
            transition={{ duration: 0.35, delay, ease: 'easeOut' }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// --- Glow pulse (for HUD elements) ---
export function GlowPulse({ children, color = 'var(--hud-cyan)', className = '' }) {
    return (
        <motion.div
            animate={{
                boxShadow: [
                    `0 0 4px ${color}20`,
                    `0 0 12px ${color}40`,
                    `0 0 4px ${color}20`,
                ],
            }}
            transition={{ duration: 2.5, repeat: Infinity, ease: 'easeInOut' }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// --- Number counter animation ---
export function AnimatedNumber({ value, duration = 0.8, className = '' }) {
    return (
        <motion.span
            key={value}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={className}
        >
            {value}
        </motion.span>
    );
}

// --- List item with hover lift ---
export function HoverLift({ children, className = '' }) {
    return (
        <motion.div
            whileHover={{ y: -2, transition: { duration: 0.15 } }}
            whileTap={{ scale: 0.98 }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// --- Page transition wrapper ---
export function PageTransition({ children, className = '' }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3, ease: 'easeOut' }}
            className={className}
        >
            {children}
        </motion.div>
    );
}

// Re-export AnimatePresence for convenience
export { AnimatePresence };

/**
 * Warning dialog shown before auto-logout due to inactivity
 */
import { useEffect, useState } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Clock } from '@phosphor-icons/react';

interface IdleWarningDialogProps {
  open: boolean;
  onStayActive: () => void;
  onLogout: () => void;
  timeRemaining: number; // seconds
}

export const IdleWarningDialog: React.FC<IdleWarningDialogProps> = ({
  open,
  onStayActive,
  onLogout,
  timeRemaining: initialTime,
}) => {
  const [timeRemaining, setTimeRemaining] = useState(initialTime);

  useEffect(() => {
    if (!open) {
      setTimeRemaining(initialTime);
      return;
    }

    const interval = setInterval(() => {
      setTimeRemaining((prev) => {
        if (prev <= 1) {
          clearInterval(interval);
          onLogout();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, [open, initialTime, onLogout]);

  return (
    <Dialog open={open} onOpenChange={() => {}}>
      <DialogContent className="sm:max-w-md" onPointerDownOutside={(e) => e.preventDefault()}>
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Clock size={24} className="text-amber-500" weight="duotone" />
            <DialogTitle>Session Expiring Soon</DialogTitle>
          </div>
          <DialogDescription>
            You've been inactive for a while. You'll be automatically signed out in{' '}
            <span className="font-semibold text-foreground">{timeRemaining} seconds</span> to protect your account.
          </DialogDescription>
        </DialogHeader>

        <DialogFooter className="sm:justify-between gap-2">
          <Button variant="outline" onClick={onLogout} className="flex-1">
            Sign Out Now
          </Button>
          <Button onClick={onStayActive} className="flex-1">
            Stay Signed In
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

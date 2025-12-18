/**
 * Login modal component for Firebase authentication
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Info, GithubLogo, GoogleLogo, X } from '@phosphor-icons/react';
import { useAuth } from '@/contexts/AuthContext';

interface LoginModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  required?: boolean; // If true, user cannot close modal without logging in
  onGuestContinue?: () => void;
}

export default function LoginModal({
  open,
  onOpenChange,
  required = false,
  onGuestContinue
}: LoginModalProps) {
  const { signInWithGoogle, signInWithGithub, continueAsGuest, user, loading: authLoading, isGuest } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Wait for user to be populated by onAuthStateChanged, then redirect
  // ONLY redirect if modal is open (meaning user just logged in)
  useEffect(() => {
    if (open && !authLoading && user && !isGuest) {
      // User is authenticated and data is loaded
      onOpenChange(false);

      // Smart redirect based on project count
      if (user.project_count === 0) {
        navigate('/project');  // New users → canvas
      } else {
        navigate('/dashboard');  // Returning users → dashboard
      }
    }
  }, [open, user, authLoading, isGuest, navigate, onOpenChange]);

  const handleGoogleSignIn = async () => {
    try {
      setError(null);
      // Just trigger sign-in - useEffect above will handle redirect after onAuthStateChanged completes
      await signInWithGoogle();
    } catch (err) {
      console.error('Google sign in error:', err);
      // Show the actual error message if it's an Error object, otherwise generic message
      const errorMessage = err instanceof Error ? err.message : 'Failed to sign in with Google. Please try again.';
      setError(errorMessage);
    }
  };

  const handleGithubSignIn = async () => {
    try {
      setError(null);
      // Just trigger sign-in - useEffect above will handle redirect after onAuthStateChanged completes
      await signInWithGithub();
    } catch (err) {
      console.error('GitHub sign in error:', err);
      // Show the actual error message if it's an Error object, otherwise generic message
      const errorMessage = err instanceof Error ? err.message : 'Failed to sign in with GitHub. Please try again.';
      setError(errorMessage);
    }
  };

  const handleContinueAsGuest = () => {
    continueAsGuest();
    onOpenChange(false);
    if (onGuestContinue) {
      onGuestContinue();
    }
  };

  const handleClose = () => {
    if (!required) {
      onOpenChange(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={handleClose}>
      <DialogContent
        className="sm:max-w-md"
        onPointerDownOutside={(e) => {
          if (required) {
            e.preventDefault();
          }
        }}
        onEscapeKeyDown={(e) => {
          if (required) {
            e.preventDefault();
          }
        }}
      >
        {!required && (
          <button
            onClick={handleClose}
            className="absolute right-4 top-4 rounded-sm opacity-70 ring-offset-background transition-opacity hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:pointer-events-none data-[state=open]:bg-accent data-[state=open]:text-muted-foreground"
          >
            <X className="h-4 w-4" />
            <span className="sr-only">Close</span>
          </button>
        )}

        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">Welcome to VisionForge</DialogTitle>
          <DialogDescription className="text-base">
            Sign in to save your work and access it from any device
          </DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-4 py-4">
          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {required && (
            <Alert>
              <Info className="h-4 w-4" />
              <AlertDescription>
                Authentication required to {required === true ? 'save or export your work' : 'continue'}
              </AlertDescription>
            </Alert>
          )}

          <Button
            onClick={handleGoogleSignIn}
            disabled={loading}
            size="lg"
            className="w-full gap-3 bg-white hover:bg-gray-50 text-gray-900 border border-gray-300"
          >
            <GoogleLogo className="h-5 w-5" weight="bold" />
            Sign in with Google
          </Button>

          <Button
            onClick={handleGithubSignIn}
            disabled={loading}
            size="lg"
            className="w-full gap-3 bg-gray-900 hover:bg-gray-800 text-white"
          >
            <GithubLogo className="h-5 w-5" weight="bold" />
            Sign in with GitHub
          </Button>

          {!required && (
            <>
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <span className="w-full border-t" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-background px-2 text-muted-foreground">Or</span>
                </div>
              </div>

              <Button
                onClick={handleContinueAsGuest}
                disabled={loading}
                variant="outline"
                size="lg"
                className="w-full"
              >
                Continue as Guest
              </Button>

              <p className="text-xs text-center text-muted-foreground">
                Note: Guest mode does not save your work. Sign in to save and access your projects.
              </p>
            </>
          )}
        </div>

        <div className="text-xs text-center text-muted-foreground mt-2">
          By continuing, you agree to our Terms of Service and Privacy Policy
        </div>
      </DialogContent>
    </Dialog>
  );
}

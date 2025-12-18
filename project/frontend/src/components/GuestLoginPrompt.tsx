/**
 * Guest Login Prompt Modal
 * Displayed when guests try to save/export their work
 * Encourages conversion from guest to authenticated user
 */
import { useEffect, useState } from 'react';
import { X } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'sonner';
import { useAuth } from '../contexts/AuthContext';
import { getGuestCanvasForTransfer, clearGuestCanvas } from '../lib/guestState';
import { createProject, saveArchitecture } from '../lib/projectApi';

interface GuestLoginPromptProps {
  isOpen: boolean;
  onClose: () => void;
  action?: 'save' | 'export' | 'general';
}

export const GuestLoginPrompt: React.FC<GuestLoginPromptProps> = ({
  isOpen,
  onClose,
  action = 'general',
}) => {
  const navigate = useNavigate();
  const { signInWithGoogle, signInWithGithub, user, loading, refreshUser } = useAuth();
  const [isProcessing, setIsProcessing] = useState(false);

  // Handle post-login flow when user becomes available
  useEffect(() => {
    const handlePostLoginFlow = async () => {
      if (!loading && user && !isProcessing) {
        setIsProcessing(true);

        try {
          // Check if guest has work to transfer
          const guestCanvas = getGuestCanvasForTransfer();

          if (guestCanvas && guestCanvas.nodes.length > 0) {
            // Guest has work - create a new project with their canvas
            const timestamp = new Date().toLocaleString('en-US', {
              month: 'short',
              day: 'numeric',
              hour: 'numeric',
              minute: '2-digit',
              hour12: true
            });

            const project = await createProject({
              name: `Guest Project - ${timestamp}`,
              description: 'Transferred from guest mode',
              framework: 'pytorch'
            });

            // Save the architecture
            await saveArchitecture(
              project.id,
              guestCanvas.nodes,
              guestCanvas.edges,
              guestCanvas.groupDefinitions ? new Map(guestCanvas.groupDefinitions.map((g: any) => [g.id, g])) : undefined
            );

            // Clear guest storage
            clearGuestCanvas();

            // Refresh user data to update project count
            await refreshUser();

            // Navigate to the new project
            navigate(`/project/${project.id}`);
            toast.success('Your work has been saved!');
          } else {
            // No guest work - smart redirect based on project count from AuthContext
            if (user.project_count === 0) {
              navigate('/project');  // New users → canvas
            } else {
              navigate('/dashboard');  // Returning users → dashboard
            }
          }

          onClose();
        } catch (error) {
          console.error('Error in post-login flow:', error);
          toast.error('Failed to save your work. Please try again.');
        } finally {
          setIsProcessing(false);
        }
      }
    };

    if (user) {
      handlePostLoginFlow();
    }
  }, [user, loading, isProcessing, navigate, onClose, refreshUser]);

  if (!isOpen) return null;

  const getActionMessage = () => {
    switch (action) {
      case 'save':
        return 'save your work';
      case 'export':
        return 'export your code';
      default:
        return 'access this feature';
    }
  };

  const handleGoogleSignIn = async () => {
    try {
      // Just trigger sign-in - useEffect will handle the post-login flow
      await signInWithGoogle();
    } catch (error) {
      console.error('Error signing in with Google:', error);
      toast.error('Failed to sign in');
    }
  };

  const handleGithubSignIn = async () => {
    try {
      // Just trigger sign-in - useEffect will handle the post-login flow
      await signInWithGithub();
    } catch (error) {
      console.error('Error signing in with GitHub:', error);
      toast.error('Failed to sign in');
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full p-6 relative">
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
          aria-label="Close"
        >
          <X size={20} />
        </button>

        {/* Header */}
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-2">
            Sign in to {getActionMessage()}
          </h2>
          <p className="text-gray-600">
            Create an account to save your projects, export code, and access all features.
          </p>
        </div>

        {/* Sign in buttons */}
        <div className="space-y-3 mb-6">
          <button
            onClick={handleGoogleSignIn}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-white border-2 border-gray-300 rounded-lg hover:bg-gray-50 hover:border-gray-400 transition-colors font-medium text-gray-700"
          >
            <svg className="w-5 h-5" viewBox="0 0 24 24">
              <path
                fill="#4285F4"
                d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              />
              <path
                fill="#34A853"
                d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              />
              <path
                fill="#FBBC05"
                d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              />
              <path
                fill="#EA4335"
                d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              />
            </svg>
            Continue with Google
          </button>

          <button
            onClick={handleGithubSignIn}
            className="w-full flex items-center justify-center gap-3 px-4 py-3 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors font-medium"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
              <path
                fillRule="evenodd"
                clipRule="evenodd"
                d="M12 2C6.477 2 2 6.477 2 12c0 4.42 2.865 8.17 6.839 9.49.5.092.682-.217.682-.482 0-.237-.008-.866-.013-1.7-2.782.603-3.369-1.34-3.369-1.34-.454-1.156-1.11-1.463-1.11-1.463-.908-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836c.85.004 1.705.114 2.504.336 1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.743 0 .267.18.578.688.48C19.137 20.167 22 16.418 22 12c0-5.523-4.477-10-10-10z"
              />
            </svg>
            Continue with GitHub
          </button>
        </div>

        {/* Guest mode note */}
        <div className="text-center">
          <p className="text-sm text-gray-500">
            Your current work is saved locally.{' '}
            <span className="font-medium">Sign in to save it to the cloud.</span>
          </p>
        </div>
      </div>
    </div>
  );
};

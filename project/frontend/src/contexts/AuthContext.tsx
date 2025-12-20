/**
 * Authentication context for Firebase authentication
 */
import React, { createContext, useContext, useEffect, useState } from 'react';
import {
  User as FirebaseUser,
  signInWithPopup,
  signOut as firebaseSignOut,
  onAuthStateChanged,
} from 'firebase/auth';
import { auth, googleProvider, githubProvider } from '../lib/firebase';
import { useModelBuilderStore } from '../lib/store';
import { useIdleTimeout } from '../hooks/useIdleTimeout';
import { IdleWarningDialog } from '../components/IdleWarningDialog';

// User type from our backend
interface VisionForgeUser {
  firebase_uid: string;
  email: string;
  display_name: string | null;
  avatar_url: string | null;
  auth_provider: 'google' | 'github';
  tier: 'free' | 'pro';
  project_count: number;
  created_at: string | null;
  last_login_at: string | null;
}

interface AuthContextType {
  user: VisionForgeUser | null;
  firebaseUser: FirebaseUser | null;
  loading: boolean;
  isGuest: boolean;
  signInWithGoogle: () => Promise<void>;
  signInWithGithub: () => Promise<void>;
  signOut: () => Promise<void>;
  continueAsGuest: () => void;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';
const GUEST_MODE_KEY = 'visionforge_guest_mode';

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<VisionForgeUser | null>(null);
  const [firebaseUser, setFirebaseUser] = useState<FirebaseUser | null>(null);
  const [loading, setLoading] = useState(true);
  // isGuest is true whenever there's no authenticated Firebase user
  // This ensures unauthenticated users are ALWAYS in guest mode
  const [isGuest, setIsGuest] = useState(true);
  const [showIdleWarning, setShowIdleWarning] = useState(false);

  // Verify token with backend and get user data
  const verifyToken = async (firebaseUser: FirebaseUser): Promise<VisionForgeUser | null> => {
    try {
      const token = await firebaseUser.getIdToken();

      const response = await fetch(`${API_BASE_URL}/auth/verify-token`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ token }),
      });

      if (!response.ok) {
        console.error('Failed to verify token with backend');
        return null;
      }

      const data = await response.json();
      return data.user;
    } catch (error) {
      console.error('Error verifying token:', error);
      return null;
    }
  };

  // Refresh user data from backend
  const refreshUser = async () => {
    if (!firebaseUser) return;

    try {
      const token = await firebaseUser.getIdToken();

      const response = await fetch(`${API_BASE_URL}/auth/me`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setUser(data.user);
      }
    } catch (error) {
      console.error('Error refreshing user:', error);
    }
  };

  // Listen to Firebase auth state changes
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      setFirebaseUser(firebaseUser);

      if (firebaseUser) {
        // User is authenticated - verify with backend and exit guest mode
        setIsGuest(false);
        localStorage.removeItem(GUEST_MODE_KEY); // Clear guest mode flag
        const backendUser = await verifyToken(firebaseUser);
        setUser(backendUser);
      } else {
        // No authenticated user - ALWAYS enter guest mode
        setIsGuest(true);
        localStorage.setItem(GUEST_MODE_KEY, 'true'); // Persist guest mode
        setUser(null);
      }

      setLoading(false);
    });

    return () => unsubscribe();
  }, []);

  // Session management - update session every 5 minutes when user is active
  useEffect(() => {
    if (!firebaseUser || isGuest) return;

    let retryCount = 0;
    const MAX_RETRIES = 3;

    const updateSession = async () => {
      const tryUpdate = async (): Promise<void> => {
        try {
          const token = await firebaseUser.getIdToken();
          const response = await fetch(`${API_BASE_URL}/auth/update-session`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({ minutes: 5 }), // Track 5 minutes of activity
          });

          if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
          }

          // Reset retry count on success
          retryCount = 0;
        } catch (error) {
          console.error('Error updating session:', error);

          // Retry with exponential backoff if under max retries
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            const backoffMs = 5000 * retryCount; // 5s, 10s, 15s
            console.log(`Retrying session update in ${backoffMs}ms (attempt ${retryCount}/${MAX_RETRIES})`);
            setTimeout(tryUpdate, backoffMs);
          } else {
            console.error('Max retries reached for session update');
            retryCount = 0; // Reset for next interval
          }
        }
      };

      await tryUpdate();
    };

    // Update session immediately on mount
    updateSession();

    // Then update every 5 minutes
    const intervalId = setInterval(updateSession, 5 * 60 * 1000);

    return () => clearInterval(intervalId);
  }, [firebaseUser, isGuest]);

  // Sign in with Google
  const signInWithGoogle = async () => {
    try {
      setLoading(true);
      // Just trigger Firebase auth - onAuthStateChanged will handle the rest
      await signInWithPopup(auth, googleProvider);
      // Don't call verifyToken here - it creates double verification
      // onAuthStateChanged listener will handle it automatically
    } catch (error: any) {
      console.error('Error signing in with Google:', error);
      setLoading(false); // Reset loading only on error

      // Handle account exists with different credential
      if (error.code === 'auth/account-exists-with-different-credential') {
        const email = error.customData?.email;
        throw new Error(`An account already exists with ${email}. Please sign in with your original provider (GitHub).`);
      }

      throw error;
    }
    // Don't set loading to false here - let onAuthStateChanged handle it
  };

  // Sign in with GitHub
  const signInWithGithub = async () => {
    try {
      setLoading(true);
      // Just trigger Firebase auth - onAuthStateChanged will handle the rest
      await signInWithPopup(auth, githubProvider);
      // Don't call verifyToken here - it creates double verification
      // onAuthStateChanged listener will handle it automatically
    } catch (error: any) {
      console.error('Error signing in with GitHub:', error);
      setLoading(false); // Reset loading only on error

      // Handle account exists with different credential
      if (error.code === 'auth/account-exists-with-different-credential') {
        const email = error.customData?.email;
        throw new Error(`An account already exists with ${email}. Please sign in with your original provider (Google).`);
      }

      throw error;
    }
    // Don't set loading to false here - let onAuthStateChanged handle it
  };

  // Sign out
  const signOut = async () => {
    try {
      // Update last login on backend
      if (firebaseUser) {
        const token = await firebaseUser.getIdToken();
        await fetch(`${API_BASE_URL}/auth/logout`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${token}`,
          },
        });
      }

      await firebaseSignOut(auth);
      // onAuthStateChanged will automatically set isGuest=true and clear user

      // Clear canvas state
      const { reset } = useModelBuilderStore.getState();
      reset();
    } catch (error) {
      console.error('Error signing out:', error);
      throw error;
    }
  };

  // Continue as guest (now just dismisses login modal - user is already in guest mode)
  const continueAsGuest = () => {
    // User is already in guest mode by default (no Firebase user)
    // This function now just ensures the state is correct and dismisses any prompts
    setIsGuest(true);
    localStorage.setItem(GUEST_MODE_KEY, 'true');
    setLoading(false);
  };

  // Idle timeout - auto logout after 10 minutes of inactivity
  // Warning shows after 2 minutes, logout after 10 minutes
  const { resetTimer } = useIdleTimeout({
    onIdle: async () => {
      if (!isGuest && user) {
        await signOut();
      }
    },
    onWarning: () => {
      if (!isGuest && user) {
        setShowIdleWarning(true);
      }
    },
    idleTime: 10 * 60 * 1000, // 10 minutes until logout
    warningTime: 8 * 60 * 1000, // 8 minutes before logout = warning at 2 min
  });

  const handleStayActive = () => {
    setShowIdleWarning(false);
    resetTimer();
  };

  const handleIdleLogout = async () => {
    setShowIdleWarning(false);
    await signOut();
  };

  const value: AuthContextType = {
    user,
    firebaseUser,
    loading,
    isGuest,
    signInWithGoogle,
    signInWithGithub,
    signOut,
    continueAsGuest,
    refreshUser,
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
      {/* Idle warning dialog */}
      {!isGuest && user && (
        <IdleWarningDialog
          open={showIdleWarning}
          onStayActive={handleStayActive}
          onLogout={handleIdleLogout}
          timeRemaining={480}
        />
      )}
    </AuthContext.Provider>
  );
};

/**
 * Dashboard Page
 * Displays user projects with stats, search, and filtering
 */
import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { Plus, MagnifyingGlass, SquaresFour, Funnel, Rocket, SignOut, User } from '@phosphor-icons/react';
import { useAuth } from '../contexts/AuthContext';
import { fetchProjects, ProjectResponse } from '../lib/projectApi';
import { ProjectCard } from '../components/ProjectCard';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { ThemeToggle } from '@/components/ThemeToggle';

export const Dashboard = () => {
  const navigate = useNavigate();
  const { user, refreshUser, signOut } = useAuth();
  const [projects, setProjects] = useState<ProjectResponse[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [frameworkFilter, setFrameworkFilter] = useState<'all' | 'pytorch' | 'tensorflow'>('all');
  const [avatarError, setAvatarError] = useState(false);

  // Load projects on mount
  useEffect(() => {
    loadProjects();
  }, []);

  // Reset avatar error when user changes
  useEffect(() => {
    setAvatarError(false);
  }, [user?.firebase_uid]);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const data = await fetchProjects();
      setProjects(data);
    } catch (error) {
      console.error('Failed to load projects:', error);
    } finally {
      setLoading(false);
    }
  };

  // Filter and search projects
  const filteredProjects = useMemo(() => {
    let filtered = projects;

    // Apply framework filter
    if (frameworkFilter !== 'all') {
      filtered = filtered.filter((p) => p.framework === frameworkFilter);
    }

    // Apply search query
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter(
        (p) =>
          p.name.toLowerCase().includes(query) ||
          p.description.toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [projects, frameworkFilter, searchQuery]);

  // Calculate stats
  const stats = useMemo(() => {
    const total = projects.length;
    const pytorch = projects.filter((p) => p.framework === 'pytorch').length;
    const tensorflow = projects.filter((p) => p.framework === 'tensorflow').length;

    return { total, pytorch, tensorflow };
  }, [projects]);

  const handleCreateProject = () => {
    navigate('/project');
  };

  const handleProjectDeleted = async () => {
    // Reload projects after deletion
    await loadProjects();
    // Refresh user data to update project count
    await refreshUser();
  };

  // PyTorch Logo Component - Matches official branding
  const PyTorchLogo = () => (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      {/* Flame/Torch */}
      <path d="M16 4L17 6C17 6 18 4 19 4V9L17 11V4Z" fill="#EE4C2C"/>
      {/* Outer circle */}
      <circle cx="16" cy="18" r="9" stroke="#EE4C2C" strokeWidth="2.5" fill="none"/>
      {/* Inner filled circle */}
      <circle cx="16" cy="18" r="5.5" fill="#EE4C2C"/>
    </svg>
  );

  // TensorFlow Logo Component - Matches official branding (3D cube)
  const TensorFlowLogo = () => (
    <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
      {/* 3D Cube */}
      <path d="M16 6L26 12L26 20L16 26L6 20L6 12L16 6Z" fill="#FF6F00" fillOpacity="0.15" stroke="#FF6F00" strokeWidth="2"/>
      <path d="M16 6L16 16L26 12" stroke="#FF6F00" strokeWidth="2" strokeLinejoin="round"/>
      <path d="M16 16L6 12" stroke="#FF6F00" strokeWidth="2" strokeLinejoin="round"/>
      <path d="M16 16L16 26" stroke="#FF6F00" strokeWidth="2"/>
    </svg>
  );

  return (
    <div className="min-h-screen bg-background">
      {/* Header Section */}
      <div className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {/* Welcome Message with User Menu */}
          <div className="mb-6 flex items-start justify-between">
            <div>
              <h1 className="text-3xl font-bold text-foreground">
                Welcome back, {user?.display_name || user?.email?.split('@')[0] || 'there'}!
              </h1>
              <p className="text-muted-foreground mt-1">
                Manage your neural network projects and continue where you left off
              </p>
            </div>

            {/* Theme Toggle and User Menu */}
            <div className="flex items-center gap-3">
              <ThemeToggle />
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="icon" className="rounded-full">
                  {user?.avatar_url && !avatarError ? (
                    <img
                      src={user.avatar_url}
                      alt={user.display_name || 'User'}
                      className="w-8 h-8 rounded-full"
                      onError={() => setAvatarError(true)}
                    />
                  ) : (
                    <User size={20} />
                  )}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                <DropdownMenuLabel>
                  <div className="flex flex-col space-y-1">
                    <p className="text-sm font-medium">{user?.display_name || 'User'}</p>
                    <p className="text-xs text-muted-foreground">{user?.email}</p>
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => navigate('/')}>
                  Home
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => navigate('/project')}>
                  New Project
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={signOut}
                  className="text-destructive focus:text-destructive"
                >
                  <SignOut size={16} className="mr-2" />
                  Sign Out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
            </div>
          </div>

          {/* Stats Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <Card className="p-4 bg-primary/5 border-primary/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">Total Projects</p>
                  <p className="text-3xl font-bold text-foreground">{stats.total}</p>
                </div>
                <SquaresFour className="text-primary" size={32} weight="duotone" />
              </div>
            </Card>

            <Card className="p-4 bg-orange-500/10 border-orange-500/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">PyTorch</p>
                  <p className="text-3xl font-bold text-foreground">{stats.pytorch}</p>
                </div>
                <PyTorchLogo />
              </div>
            </Card>

            <Card className="p-4 bg-amber-500/10 border-amber-500/20">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-muted-foreground">TensorFlow</p>
                  <p className="text-3xl font-bold text-foreground">{stats.tensorflow}</p>
                </div>
                <TensorFlowLogo />
              </div>
            </Card>
          </div>

          {/* Search and Filters */}
          <div className="flex flex-col sm:flex-row gap-4 items-stretch sm:items-center">
            {/* Search Bar */}
            <div className="flex-1 relative">
              <MagnifyingGlass className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground" size={20} />
              <Input
                type="text"
                placeholder="Search projects by name or description..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10"
              />
            </div>

            {/* Framework Filter */}
            <Select value={frameworkFilter} onValueChange={(value) => setFrameworkFilter(value as 'all' | 'pytorch' | 'tensorflow')}>
              <SelectTrigger className="w-[180px]">
                <div className="flex items-center gap-2">
                  <Funnel size={16} />
                  <SelectValue placeholder="All Frameworks" />
                </div>
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Frameworks</SelectItem>
                <SelectItem value="pytorch">PyTorch</SelectItem>
                <SelectItem value="tensorflow">TensorFlow</SelectItem>
              </SelectContent>
            </Select>

            {/* Create Button */}
            <Button onClick={handleCreateProject} className="whitespace-nowrap">
              <Plus size={20} />
              New Project
            </Button>
          </div>
        </div>
      </div>

      {/* Projects Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {loading ? (
          // Loading State
          <div className="flex flex-col items-center justify-center py-12">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
            <p className="text-muted-foreground">Loading your projects...</p>
          </div>
        ) : filteredProjects.length === 0 ? (
          // Empty State
          <div className="flex flex-col items-center justify-center py-12">
            {searchQuery || frameworkFilter !== 'all' ? (
              // No search results
              <>
                <MagnifyingGlass size={64} className="text-muted-foreground mb-4" weight="duotone" />
                <h3 className="text-xl font-semibold text-foreground mb-2">No projects found</h3>
                <p className="text-muted-foreground mb-6">
                  Try adjusting your search or filter criteria
                </p>
                <Button
                  variant="secondary"
                  onClick={() => {
                    setSearchQuery('');
                    setFrameworkFilter('all');
                  }}
                >
                  Clear filters
                </Button>
              </>
            ) : (
              // No projects at all
              <>
                <Rocket size={64} className="text-muted-foreground mb-4" weight="duotone" />
                <h3 className="text-xl font-semibold text-foreground mb-2">No projects yet</h3>
                <p className="text-muted-foreground mb-6">
                  Create your first neural network project to get started
                </p>
                <Button onClick={handleCreateProject} size="lg">
                  <Plus size={20} />
                  Create Your First Project
                </Button>
              </>
            )}
          </div>
        ) : (
          // Projects Grid
          <>
            <div className="mb-4">
              <p className="text-muted-foreground">
                Showing {filteredProjects.length} of {projects.length} project{projects.length !== 1 ? 's' : ''}
              </p>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredProjects.map((project) => (
                <ProjectCard
                  key={project.id}
                  project={project}
                  onDeleted={handleProjectDeleted}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

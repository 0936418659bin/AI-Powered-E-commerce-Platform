"use client";

import React from "react";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export function ErrorBoundary({ children, fallback }: ErrorBoundaryProps) {
  const [hasError, setHasError] = React.useState(false);

  React.useEffect(() => {
    setHasError(false);
  }, [children]);

  if (hasError) {
    return (
      fallback || (
        <div className="p-4 border border-red-200 rounded-lg bg-red-50 text-red-700">
          <p className="text-sm">
            Something went wrong loading this component.
          </p>
        </div>
      )
    );
  }

  try {
    return <>{children}</>;
  } catch (error) {
    setHasError(true);
    return (
      fallback || (
        <div className="p-4 border border-red-200 rounded-lg bg-red-50 text-red-700">
          <p className="text-sm">
            Something went wrong loading this component.
          </p>
        </div>
      )
    );
  }
}
